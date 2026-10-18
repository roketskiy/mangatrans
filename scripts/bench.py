"""Reproducible Windows end-to-end file benchmark. See docs/performance.md."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import cmp_to_key
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zipfile

import psutil
from PIL import Image

from winjob import Job

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=h.hexdigest())


def check_identity(path, expected):
    try:
        actual = identity(path)
        return dict(unchanged=actual == expected, actual=actual)
    except OSError as e:
        return dict(unchanged=False, error=f"{type(e).__name__}: {e}")


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    # Nearest-rank p95; small sample sizes are explicitly visible through n.
    return dict(n=len(values), min=ordered[0], median=statistics.median(ordered),
                p95=ordered[math.ceil(.95 * len(values)) - 1], max=ordered[-1],
                mean=statistics.mean(ordered),
                stdev=statistics.stdev(ordered) if len(ordered) > 1 else None)


def natural_compare(a, b):
    # Match internal/cbz.naturalLess, including leading-zero ties.
    i = j = 0
    while i < len(a) and j < len(b):
        if "0" <= a[i] <= "9" and "0" <= b[j] <= "9":
            ia, jb = i, j
            while i < len(a) and "0" <= a[i] <= "9":
                i += 1
            while j < len(b) and "0" <= b[j] <= "9":
                j += 1
            aa, bb = a[ia:i].lstrip("0"), b[jb:j].lstrip("0")
            if len(aa) != len(bb):
                return (len(aa) > len(bb)) - (len(aa) < len(bb))
            if aa != bb:
                return (aa > bb) - (aa < bb)
        else:
            if a[i] != b[j]:
                return (a[i] > b[j]) - (a[i] < b[j])
            i, j = i + 1, j + 1
    return (len(a) - i > len(b) - j) - (len(a) - i < len(b) - j)


def archive_entries(z):
    return [e for e in z.infolist()
            if not e.is_dir() and Path(e.filename).suffix.lower() in EXTENSIONS]


def validate_output(source, output, limit, trace):
    with zipfile.ZipFile(source) as src, zipfile.ZipFile(output) as dst:
        inputs = sorted(archive_entries(src),
                        key=cmp_to_key(lambda a, b: natural_compare(a.filename, b.filename)))
        if limit:
            inputs = inputs[:limit]
        outputs = archive_entries(dst)
        names = [e.filename for e in inputs]
        if not names or names != [e.filename for e in outputs]:
            raise ValueError("Output page names/order/count do not match selected input pages")
        if names != [p["name"] for p in trace["pages"]]:
            raise ValueError("Trace pages do not match selected input pages")
        pages = []
        for before, after, metric in zip(inputs, outputs, trace["pages"]):
            original, data = src.read(before), dst.read(after)  # Checks ZIP CRC too.
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                size = list(image.size)
            same = original == data
            if metric["outcome"] != "rendered" and not same:
                raise ValueError(f"Unexpected change in retained page: {after.filename}")
            if metric["outcome"] == "rendered" and (same or metric["rendered"] <= 0):
                raise ValueError(f"Rendered page has no observable work: {after.filename}")
            if metric["width"] and size != [metric["width"], metric["height"]]:
                raise ValueError(f"Dimensions changed: {after.filename}")
            pages.append(dict(name=after.filename, dimensions=size, bytes=len(data),
                              unchanged=same, sha256=hashlib.sha256(data).hexdigest()))
    return dict(valid=True, pages=pages, changed_pages=sum(not p["unchanged"] for p in pages))


def analyze_trace(trace):
    if trace.get("schema") != 1 or trace.get("total_ns", 0) <= 0:
        raise ValueError("Missing/unsupported -perf-json telemetry; rebuild the engine")
    groups = defaultdict(list)
    counters = defaultdict(Counter)
    intervals = []
    for e in trace["events"]:
        start, duration = e["start_ns"], e["duration_ns"]
        if start < 0 or duration < 0 or start + duration > trace["total_ns"]:
            raise ValueError("Invalid telemetry interval")
        if e["kind"] == "counter":
            if duration != 0 or not isinstance(e.get("count"), int) or e["count"] < 0:
                raise ValueError("Invalid telemetry counter")
            counters[e["stage"]][e.get("page", -1)] += e["count"]
            continue
        groups[(e["kind"], e["stage"])].append(e)
        if e["kind"] == "operation":
            intervals.append((start, start + duration))
    covered, end = 0, 0
    for left, right in sorted(intervals):
        covered += max(0, right - max(left, end))
        end = max(end, right)
    stages = []
    for (kind, stage), events in sorted(groups.items()):
        values = [e["duration_ns"] / 1e6 for e in events]
        stages.append(dict(kind=kind, stage=stage, total_ms=sum(values),
                           errors=sum(bool(e.get("error")) for e in events),
                           latency_ms=distribution(values)))
    pages = trace["pages"]
    per_page = defaultdict(lambda: defaultdict(float))
    per_page_intervals = defaultdict(list)
    for event in trace["events"]:
        if event["kind"] == "operation" and event["page"] >= 0:
            per_page[event["page"]][event["stage"]] += event["duration_ns"] / 1e6
            per_page_intervals[event["page"]].append((event["start_ns"], event["start_ns"] + event["duration_ns"]))
    page_operations = []
    for page in pages:
        per_stage = per_page[page["index"]]
        spans = sorted(per_page_intervals[page["index"]])
        measured, end = 0, 0
        for left, right in spans:
            measured += max(0, right - max(left, end))
            end = max(end, right)
        page_operations.append(page | dict(operations_ms=dict(per_stage), measured_ms=measured / 1e6,
                                           counters={name: counts.get(page["index"], 0) for name, counts in counters.items()}))
    return dict(total_ms=trace["total_ns"] / 1e6, covered_ms=covered / 1e6,
                uninstrumented_ms=(trace["total_ns"] - covered) / 1e6,
                stages=stages, counters={name: dict(total=sum(counts.values()), pages={str(page): count for page, count in sorted(counts.items()) if page >= 0}, unassigned=counts.get(-1, 0)) for name, counts in counters.items()}, page_operations=page_operations,
                outcomes=dict(Counter(p["outcome"] for p in pages)),
                pages=len(pages), detected=sum(p["detected"] for p in pages),
                ocr_accepted=sum(p["ocr_accepted"] for p in pages),
                ocr_failed=sum(p.get("ocr_failed", 0) for p in pages),
                ocr_rejected=sum(p.get("ocr_rejected", 0) for p in pages),
                crop_skipped=sum(p.get("crop_skipped", 0) for p in pages),
                translated=sum(p["translated"] for p in pages),
                rendered=sum(p["rendered"] for p in pages),
                untranslated=sum(p["ocr_accepted"] - p["translated"] for p in pages),
                operation_errors=sum(s["errors"] for s in stages if s["kind"] == "operation"),
                request_errors=sum(s["errors"] for s in stages if s["kind"] == "request"))


class GPUCollector:
    """A separate thread prevents nvidia-smi latency from stalling process sampling."""
    def __init__(self, interval):
        self.interval = interval
        self.stop = threading.Event()
        self.rows, self.errors = [], []
        self.exe = shutil.which("nvidia-smi")
        self.thread = None

    def query(self):
        fields = "index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu"
        begin = time.perf_counter_ns()
        result = subprocess.run([self.exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, creationflags=NO_WINDOW)
        end = time.perf_counter_ns()
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        rows = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 10:
                raise ValueError(f"Unexpected nvidia-smi row: {row}")
            values = {}
            for key, value in zip(("total_mib", "used_mib", "gpu_percent", "memory_percent",
                                   "power_w", "temperature_c"), row[4:]):
                try:
                    values[key] = float(value)
                except ValueError:
                    values[key] = None
            rows.append(dict(query_start_ns=begin, query_end_ns=end, index=row[0],
                             uuid=row[1], name=row[2], driver=row[3], **values))
        return rows

    def start(self):
        if not self.exe:
            self.errors.append("nvidia-smi unavailable; GPU metrics are null")
            return
        def worker():
            while not self.stop.is_set():
                begin = time.perf_counter()
                try:
                    self.rows.extend(self.query())
                except Exception as e:
                    self.errors.append(str(e))
                self.stop.wait(max(0, self.interval - (time.perf_counter() - begin)))
        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=6)


def run_process(command, directory, interval, timeout, gpu_interval):
    job = Job()
    gpu = GPUCollector(gpu_interval)
    rows, monitor_errors, known = [], Counter(), {}
    timed_out = False
    interrupted = False
    # Baseline GPU queries and host load measurement are outside the timed region.
    baseline_gpu = None
    if gpu.exe:
        try:
            baseline_gpu = gpu.query()
        except Exception as e:
            gpu.errors.append(str(e))
    host_cpu_before = psutil.cpu_percent(interval=.25)
    gpu.start()
    proc = None
    try:
        with (directory / "engine.log").open("wb") as log, (directory / "samples.jsonl").open("w", encoding="utf-8") as raw:
            start = time.perf_counter_ns()
            proc = job.launch(command, log, ROOT)
            try:
                root = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                root = None
            root_exit_seen = None
            last_active_ns = start
            while True:
                tick = time.perf_counter_ns()
                alive = proc.poll() is None
                if alive and root is not None:
                    try:
                        for p in [root, *root.children(recursive=True)]:
                            known[(p.pid, p.create_time())] = p
                    except psutil.Error as e:
                        monitor_errors[type(e).__name__] += 1
                processes = []
                for key, p in list(known.items()):
                    try:
                        with p.oneshot():
                            mem, cpu, disk = p.memory_info(), p.cpu_times(), p.io_counters()
                            processes.append(dict(pid=p.pid, created=key[1], name=p.name(),
                                                  rss_bytes=mem.rss, private_bytes=getattr(mem, "private", None),
                                                  cpu_s=cpu.user + cpu.system,
                                                  io_read_bytes=disk.read_bytes, io_write_bytes=disk.write_bytes))
                    except psutil.NoSuchProcess:
                        del known[key]
                    except psutil.Error as e:
                        monitor_errors[type(e).__name__] += 1
                account = job.totals()
                if account["active_processes"]:
                    last_active_ns = tick
                row = dict(elapsed_s=(tick - start) / 1e9,
                           tree_rss_bytes=sum(p["rss_bytes"] for p in processes),
                           system_cpu_percent=psutil.cpu_percent(),
                           system_available_bytes=psutil.virtual_memory().available,
                           job=account, processes=processes)
                row["collection_ms"] = (time.perf_counter_ns() - tick) / 1e6
                rows.append(row)
                raw.write(json.dumps(row) + "\n")
                if not alive and root_exit_seen is None:
                    root_exit_seen = time.perf_counter()
                if not alive and (account["active_processes"] == 0 or time.perf_counter() - root_exit_seen >= 2):
                    finish = time.perf_counter_ns()
                    break
                if not timed_out and (tick - start) / 1e9 >= timeout:
                    timed_out = True
                    job.terminate()
                    proc.wait(timeout=10)
                time.sleep(max(0, interval - (time.perf_counter_ns() - tick) / 1e9))
            raw.flush()
            totals = job.totals()
            # Root exit with live descendants is a failure; terminate them before returning.
            residual_children = totals["active_processes"]
            if residual_children:
                job.terminate()
            wall_s = (finish - start) / 1e9
    except KeyboardInterrupt:
        interrupted = True
        if proc:
            job.terminate()
            proc.wait(timeout=10)
        finish = time.perf_counter_ns()
        wall_s = (finish - start) / 1e9
        totals = job.totals()
        residual_children = 0
    finally:
        gpu.finish()
        job.close()
        if proc:
            proc.wait(timeout=10)
    active_gpu = []
    for row in gpu.rows:
        # Exclude readings whose query interval extends outside the process interval.
        if start <= row["query_start_ns"] and row["query_end_ns"] <= finish:
            active_gpu.append(dict(elapsed_s=(row["query_end_ns"] - start) / 1e9,
                                   query_duration_ms=(row["query_end_ns"] - row["query_start_ns"]) / 1e6,
                                   **row))
    dump(directory / "gpu.json", dict(baseline=baseline_gpu, samples=active_gpu, errors=gpu.errors))
    gaps = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(rows, rows[1:])]
    return dict(exit_code=proc.returncode if proc else None, timed_out=timed_out, interrupted=interrupted,
                clock=dict(start_counter_ns=start, finish_counter_ns=finish),
                residual_children=residual_children, wall_s=wall_s,
                wall_observation_uncertainty_s=(finish - last_active_ns) / 1e9,
                resources=dict(job=totals, average_cpu_cores=totals["cpu_s"] / wall_s,
                               average_cpu_machine_percent=100 * totals["cpu_s"] / wall_s / psutil.cpu_count(),
                               sampled_tree_rss_peak_bytes=max((r["tree_rss_bytes"] for r in rows), default=None),
                               sample_count=len(rows), sample_gap_s=distribution(gaps),
                               collection_ms=distribution([r["collection_ms"] for r in rows]),
                               monitor_errors=dict(monitor_errors), host_cpu_before_percent=host_cpu_before),
                gpu=dict(samples=len(active_gpu), errors=gpu.errors,
                         devices={uuid: {key: distribution([r[key] for r in active_gpu
                                                           if r["uuid"] == uuid and r[key] is not None])
                                         for key in ("used_mib", "gpu_percent", "memory_percent", "power_w", "temperature_c")}
                                  for uuid in sorted({r["uuid"] for r in active_gpu})}))


def align_resources(result, trace, directory):
    anchor = trace.get("clock")
    if not anchor:
        return dict(available=False, reason="Engine did not supply a QPC clock anchor")
    offset = anchor["qpc_ns"] - anchor["go_ns"]
    clock = result["clock"]
    uncertainty = anchor["uncertainty_ns"]
    if not (clock["start_counter_ns"] - uncertainty <= offset <= clock["finish_counter_ns"]):
        return dict(available=False, reason="QPC anchor is incompatible with Python's clock")
    samples = [json.loads(line) for line in (directory / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    gpu = json.loads((directory / "gpu.json").read_text(encoding="utf-8"))["samples"]
    stages = defaultdict(lambda: dict(rss=[], cpu_s=0., cpu_window_s=0., gpu=defaultdict(list)))
    for event in trace["events"]:
        if event["kind"] != "operation":
            continue
        # Shrink windows by the clock anchor's uncertainty.
        left = offset + event["start_ns"] + uncertainty
        right = offset + event["start_ns"] + event["duration_ns"] - uncertainty
        row = stages[event["stage"]]
        for sample in samples:
            point = clock["start_counter_ns"] + int(sample["elapsed_s"] * 1e9)
            end = point + int(sample["collection_ms"] * 1e6)
            if left <= point and end <= right:
                row["rss"].append(sample["tree_rss_bytes"])
        for a, b in zip(samples, samples[1:]):
            # Account query happens during collection, not exactly at its start.
            a_begin = clock["start_counter_ns"] + int(a["elapsed_s"] * 1e9)
            b_end = clock["start_counter_ns"] + int((b["elapsed_s"] + b["collection_ms"] / 1000) * 1e9)
            if left <= a_begin and b_end <= right:
                row["cpu_s"] += max(0, b["job"]["cpu_s"] - a["job"]["cpu_s"])
                row["cpu_window_s"] += b["elapsed_s"] - a["elapsed_s"]
        for sample in gpu:
            if left <= sample["query_start_ns"] and sample["query_end_ns"] <= right:
                row["gpu"][sample["uuid"]].append(sample)
    return dict(available=True, clock_uncertainty_ns=uncertainty, stages={
        stage: dict(rss_samples=len(row["rss"]), sampled_rss_peak_bytes=max(row["rss"], default=None),
                    cpu_s_inside_sample_windows=row["cpu_s"] if row["cpu_window_s"] else None,
                    cpu_sample_window_s=row["cpu_window_s"],
                    gpu={uuid: dict(samples=len(values),
                                    sampled_used_mib_peak=max((s["used_mib"] for s in values if s["used_mib"] is not None), default=None),
                                    utilization_percent=distribution([s["gpu_percent"] for s in values if s["gpu_percent"] is not None]))
                         for uuid, values in row["gpu"].items()})
        for stage, row in sorted(stages.items())})


def evaluate(result, source, output, limit, trace_path):
    issues = []
    if result["exit_code"] != 0:
        issues.append(f"engine exit code {result['exit_code']}")
    for key in ("timed_out", "interrupted", "residual_children"):
        if result[key]:
            issues.append(key)
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        analysis = analyze_trace(trace)
        analysis["external_minus_engine_ms"] = result["wall_s"] * 1000 - analysis["total_ms"]
        result["analysis"] = analysis
        result["stage_resources"] = align_resources(result, trace, trace_path.parent)
        if trace.get("error"):
            issues.append(trace["error"])
        if not issues:
            result["validation"] = validate_output(source, output, limit, trace)
        if any(k.endswith("_failed") or k == "unprocessed" for k in analysis["outcomes"]):
            issues.append("one or more pages failed or were not processed")
        if analysis["ocr_failed"] or analysis["untranslated"]:
            issues.append("failed OCR or untranslated accepted OCR blocks")
        if any(s["errors"] and s["stage"].startswith("cleanup_") for s in analysis["stages"]):
            issues.append("resource cleanup errors")
        if not analysis["rendered"]:
            issues.append("no text blocks were rendered; this is not a completed translation workload")
    except Exception as e:
        issues.append(f"telemetry/output validation: {type(e).__name__}: {e}")
    result["issues"] = issues
    recovered = result.get("analysis", {}).get("operation_errors", 0) + result.get("analysis", {}).get("request_errors", 0)
    result["status"] = "failed_or_degraded" if issues else "ok_with_recovered_errors" if recovered else "ok"
    return result


def summarize(runs):
    measured = [r for r in runs if not r["warmup"]]
    good = [r for r in measured if r["status"] in ("ok", "ok_with_recovered_errors")]
    stages = defaultdict(list)
    for r in good:
        for s in r["analysis"]["stages"]:
            if s["kind"] == "operation":
                stages[s["stage"]].append(s["total_ms"])
    return dict(measured=len(measured), successful=len(good),
                failed_or_degraded=len(measured) - len(good),
                successful_with_recovered_errors=sum(r["status"] == "ok_with_recovered_errors" for r in good),
                wall_s=distribution([r["wall_s"] for r in good]),
                pages_per_minute=distribution([r["analysis"]["pages"] * 60 / r["wall_s"] for r in good]),
                stages_ms={k: distribution(v) for k, v in sorted(stages.items())})


def markdown(report):
    summary = report["summary"]
    lines = ["# 文件处理性能测试", "", f"输入：`{report['input']['path']}`", "",
             f"正式轮次 {summary['measured']}，成功 {summary['successful']}，失败或降级 {summary['failed_or_degraded']}。",
             f"成功轮次中有 {summary['successful_with_recovered_errors']} 轮含已恢复错误，其重试耗时保留在汇总中。",
             "仅完整成功轮次进入汇总；失败与降级的耗时仍保留在逐轮记录中。", "",
             "| 轮次 | 用途 | 状态 | 总耗时 s | CPU s | 平均占用核数 | 采样 RSS 峰值 MiB | Job 提交内存峰值 MiB |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    if "integrity" in report:
        lines.insert(4, f"输入与程序完整性检查：`{json.dumps(report['integrity'], ensure_ascii=False)}`")
    for r in report["runs"]:
        resource = r["resources"]
        lines.append(f"| {r['index']} | {'预热' if r['warmup'] else '正式'} | {r['status']} | "
                     f"{r['wall_s']:.3f} | {resource['job']['cpu_s']:.3f} | {resource['average_cpu_cores']:.2f} | "
                     f"{(resource['sampled_tree_rss_peak_bytes'] or 0)/2**20:.1f} | "
                     f"{resource['job']['peak_job_commit_bytes']/2**20:.1f} |")
    if summary["wall_s"]:
        d = summary["wall_s"]
        lines += ["", f"成功轮次总耗时：中位数 {d['median']:.3f} s，范围 {d['min']:.3f}–{d['max']:.3f} s，"
                  f"p95 {d['p95']:.3f} s（n={d['n']}，nearest-rank）。"]
    lines += ["", "| 阶段（每轮累计） | 中位数 ms | 最小 ms | 最大 ms |", "| --- | ---: | ---: | ---: |"]
    for name, d in summary["stages_ms"].items():
        lines.append(f"| {name} | {d['median']:.3f} | {d['min']:.3f} | {d['max']:.3f} |")
    for r in report["runs"]:
        a = r.get("analysis")
        lines += ["", f"## 第 {r['index']} 轮", "", f"诊断：{'; '.join(r['issues']) or '通过'}"]
        if a:
            lines += ["", f"页结果：`{json.dumps(a['outcomes'], ensure_ascii=False)}`；"
                      f"检测框 {a['detected']}，有效 OCR {a['ocr_accepted']}，译文块 {a['translated']}，"
                      f"实际绘制块 {a['rendered']}，未译块 {a['untranslated']}。",
                      f"OCR 失败 {a['ocr_failed']}，空白/噪声识别 {a['ocr_rejected']}，无效裁剪 {a['crop_skipped']}。",
                      f"引擎计时 {a['total_ms']:.3f} ms，未埋点 {a['uninstrumented_ms']:.3f} ms，"
                      f"外部与引擎之差 {a['external_minus_engine_ms']:.3f} ms（启动、遥测写盘及退出观察等）。",
                      "", "| 类型 | 阶段 | 次数 | 总 ms | 中位数 ms | p95 ms | 错误数 |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for s in a["stages"]:
                d = s["latency_ms"]
                lines.append(f"| {s['kind']} | {s['stage']} | {d['n']} | {s['total_ms']:.3f} | "
                             f"{d['median']:.3f} | {d['p95']:.3f} | {s['errors']} |")
            lines += ["", "| 页（从 1 开始） | 尺寸 | 结果 | 检测框 | 有效 OCR | 译文块 | 绘制块 | 已埋点操作 ms |",
                      "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for p in a["page_operations"]:
                lines.append(f"| {p['index']+1} | {p['width']}×{p['height']} | {p['outcome']} | "
                             f"{p['detected']} | {p['ocr_accepted']} | {p['translated']} | "
                             f"{p['rendered']} | {p['measured_ms']:.3f} |")
        lines += ["", f"Job CPU / I/O：`{json.dumps(r['resources']['job'])}`",
                  "", f"GPU（整卡，包含其他应用）：`{json.dumps(r['gpu'], ensure_ascii=False)}`",
                  "", f"采样间隔：`{json.dumps(r['resources']['sample_gap_s'])}` s；"
                  f"采样异常：`{json.dumps(r['resources']['monitor_errors'])}`。"]
        aligned = r.get("stage_resources", {})
        if aligned.get("available"):
            lines += ["", f"阶段资源（QPC 对齐范围误差 {aligned['clock_uncertainty_ns']} ns，另有采样粒度）：",
                      "", "| 阶段 | RSS 样本数 | 采样 RSS 峰值 MiB | 完全落在阶段内的 CPU 采样窗口 s | 这些窗口内的 CPU s |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for stage, s in aligned["stages"].items():
                rss = f"{s['sampled_rss_peak_bytes']/2**20:.1f}" if s['sampled_rss_peak_bytes'] is not None else "—"
                cpu = f"{s['cpu_s_inside_sample_windows']:.3f}" if s['cpu_s_inside_sample_windows'] is not None else "—"
                lines.append(f"| {stage} | {s['rss_samples']} | {rss} | {s['cpu_sample_window_s']:.3f} | {cpu} |")
        else:
            lines += ["", f"阶段资源未对齐：{aligned.get('reason', '缺少采样')}。"]
    lines += ["", "## 测量边界", "", *[f"- {note}" for note in report["methodology"]], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Any input CBZ/ZIP")
    parser.add_argument("--exe", type=Path, default=ROOT / "bin/mangatrans.exe")
    parser.add_argument("--output-dir", type=Path, help="New directory; must not exist")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--interval", type=float, default=.1, help="Process sampling interval, seconds")
    parser.add_argument("--gpu-interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=3600, help="Per-run timeout, seconds")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--engine-arg", action="append", default=[], help="One extra argument token; use --engine-arg=-font")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This application and its Job Object resource accounting require Windows")
    if args.runs < 1 or args.warmup < 0 or args.limit < 0:
        parser.error("runs >= 1, warmup >= 0, limit >= 0 required")
    if not all(math.isfinite(x) and x > 0 for x in (args.interval, args.gpu_interval, args.timeout)):
        parser.error("intervals and timeout must be finite and positive")
    if args.interval < .02 or args.gpu_interval < .2:
        parser.error("Minimum process interval is .02 s; minimum GPU interval is .2 s")
    protected = {"-in", "-out", "-perf-json", "-bench-json", "-limit", "-work"}
    if any(token.lstrip("-").split("=")[0] in {p.lstrip("-") for p in protected}
           for token in args.engine_arg):
        parser.error("Input/output/telemetry/limit/work flags are controlled by the benchmark")
    source, exe = args.input.resolve(strict=True), args.exe.resolve(strict=True)
    if not source.is_file() or not exe.is_file():
        parser.error("input and exe must be files")
    help_result = subprocess.run([str(exe), "-h"], capture_output=True, text=True,
                                 creationflags=NO_WINDOW, timeout=15)
    if "-perf-json" not in help_result.stdout + help_result.stderr:
        parser.error("Engine lacks new telemetry; rebuild cmd/mangatrans with -perf-json support")
    out = (args.output_dir or ROOT / ".cache" / "bench" /
           datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema=1, started_utc=datetime.now(timezone.utc).isoformat(),
                  input=identity(source), executable=identity(exe),
                  measurement_scripts={name: identity(ROOT / "scripts" / name) for name in ("bench.py", "winjob.py")},
                  host=dict(platform=platform.platform(), python=sys.version, psutil=psutil.__version__,
                            logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False),
                            memory_bytes=psutil.virtual_memory().total),
                  settings=vars(args) | {"input": str(source), "exe": str(exe), "output_dir": str(out)},
                  environment={k: v for k, v in os.environ.items() if k.startswith("MANGATRANS_")},
                  methodology=[
                      "仅依据新 -perf-json 埋点；不读取旧 bench-json 或从文本日志推断阶段。",
                      "Windows 埋点直接读取 QPC 高精度单调计数器，保留纳秒换算原值和计数频率；纳秒字段不代表纳秒准确度。",
                      "外部墙钟覆盖启动、处理、清理、遥测写盘及退出；退出通过轮询观察，上界误差见 wall_observation_uncertainty_s。校验、散列和预检在计时之外。",
                      "操作耗时是主机观察到的同步调用延迟（包含预处理、传输、等待等），不是独立 CUDA kernel 时长。",
                      "operation 阶段覆盖时间按区间并集计算；nested 校验和 request 请求嵌套于操作，不能再次相加。未埋点时间单列。",
                      "Windows Job Object 在挂起的引擎恢复执行前绑定；CPU、I/O、提交内存峰值包含已退出子进程。I/O 是进程操作计数，包含缓存及设备操作，不等于物理磁盘流量。",
                      "RSS 为同轮依次观测的进程工作集之和，非原子快照；共享页可能重复计算，瞬时峰值可能漏采。Job 提交内存峰值与 RSS 含义不同。",
                      "GPU 为每张整卡的 nvidia-smi 采样，包含其他应用；不把整卡显存当作本应用独占显存。无数据为 null/空并附错误原因。",
                      "GPU 查询单独线程运行且有超时；查询起止和采集开销保留。采样会产生扰动，未声称零开销。",
                      "阶段资源使用同一 Windows QPC 时钟，时间粒度由频率决定。仅采用完全落入阶段内的采集窗口，边界窗口舍弃；阶段 CPU 只是已覆盖窗口的累计，不冒充阶段完整 CPU。短阶段无样本时留空。GPU 利用率本身还含驱动的历史采样窗口。",
                      "每轮新进程，保留首轮；warmup 仅排除汇总，不意味着清空或控制了 OS 文件缓存。输入预检散列会预热文件缓存。",
                      "失败、重试、页回退、有效 OCR 未译及零绘制均可见；降级不进成功汇总，恢复成功的重试连同其耗时进入汇总，避免只挑快样本。",
                      "校验页名、顺序、数量、CRC、全图解码、尺寸及实际字节变化；不能由此证明翻译语义或视觉排版正确。",
                      "CPU/GPU 占用与阶段延迟提供瓶颈线索，不直接等同于因果剖析或优化收益；少量轮次的 p95 仅作描述。",
                  ], runs=[])
    # Convert pathlib values before the first checkpoint.
    report["settings"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["settings"].items()}
    report["summary"] = summarize([])
    dump(out / "report.json", report)
    for i in range(args.warmup + args.runs):
        directory = out / f"run-{i+1:03d}"
        directory.mkdir()
        output, trace = directory / "output.cbz", directory / "trace.json"
        command = [str(exe), "-in", str(source), "-out", str(output), "-perf-json", str(trace),
                   "-limit", str(args.limit), *args.engine_arg]
        print(f"[{i+1}/{args.warmup + args.runs}] {source.name}", flush=True)
        dump(directory / "command.json", command)
        try:
            result = run_process(command, directory, args.interval, args.timeout, args.gpu_interval)
            evaluate(result, source, output, args.limit, trace)
        except Exception as e:
            # Infrastructure failures are distinct from application failures.
            dump(directory / "infrastructure-error.json", dict(error=f"{type(e).__name__}: {e}"))
            print(f"Benchmark infrastructure failed: {e}. Evidence: {directory}", file=sys.stderr)
            return 2
        result.update(index=i+1, warmup=i < args.warmup, directory=str(directory), command=command)
        dump(directory / "result.json", result)
        report["runs"].append(result)
        report["summary"] = summarize(report["runs"])
        dump(out / "report.json", report)
        (out / "report.md").write_text(markdown(report), encoding="utf-8")
        print(f"  {result['status']}: {result['wall_s']:.3f}s; {directory}", flush=True)
        if result["interrupted"]:
            return 130
    report["integrity"] = dict(input=check_identity(source, report["input"]),
                               executable=check_identity(exe, report["executable"]))
    report["input_unchanged"] = report["integrity"]["input"]["unchanged"]
    report["executable_unchanged"] = report["integrity"]["executable"]["unchanged"]
    dump(out / "report.json", report)
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    print(f"Report: {out / 'report.md'}")
    return 0 if report["summary"]["failed_or_degraded"] == 0 and report["input_unchanged"] and report["executable_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
