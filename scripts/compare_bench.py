"""Paired A/B benchmark runner. Each engine measurement is delegated to bench.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile
from functools import cmp_to_key

import bench


def digest(path):
    return bench.identity(path)


def dist(values):
    if not values:
        return None
    x = sorted(values)
    return dict(n=len(x), median=statistics.median(x), p95=x[int((len(x) * 95 + 99) // 100) - 1],
                minimum=x[0], maximum=x[-1], range=x[-1] - x[0])


def workload(run):
    analysis = run.get("analysis", {})
    return [(p.get("name"), p.get("width"), p.get("height"), p.get("detected"),
             p.get("ocr_accepted"), p.get("ocr_failed", 0), p.get("ocr_rejected", 0),
             p.get("translated"), p.get("rendered"), p.get("outcome"))
            for p in analysis.get("page_operations", [])]


def workload_is_comparable(run):
    pages = workload(run)
    analysis_pages = run.get("analysis", {}).get("pages")
    validated_pages = run.get("validation", {}).get("pages")
    return bool(pages) and analysis_pages == len(pages) and isinstance(validated_pages, list) and len(validated_pages) == len(pages)


def selected_page_count(source, limit):
    with zipfile.ZipFile(source) as archive:
        pages = sorted(bench.archive_entries(archive),
                       key=cmp_to_key(lambda a, b: bench.natural_compare(a.filename, b.filename)))
    return min(len(pages), limit) if limit else len(pages)


def completed_result(report):
    """Return the single finished run or reject an incomplete bench checkpoint."""
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 1 or "integrity" not in report:
        raise ValueError("existing bench output is incomplete; inspect it before resuming")
    result = runs[0]
    result["integrity_unchanged"] = bool(report.get("input_unchanged") and report.get("executable_unchanged"))
    return result


def engine_arg_tokens(values):
    return ["--engine-arg=" + token for token in values]


def resources(run):
    res = run.get("resources", {})
    job = res.get("job", {})
    gpu = run.get("gpu", {}).get("devices", {})
    return {"wall_s": run.get("wall_s"), "rss_peak_bytes": res.get("sampled_tree_rss_peak_bytes"),
            "job_commit_peak_bytes": job.get("peak_job_commit_bytes"), "cpu_s": job.get("cpu_s"),
            "gpu_memory_peak_mib": {uuid: (values.get("used_mib") or {}).get("max")
                                    for uuid, values in gpu.items()}}


def comparison(pairs, expected_pages=None, include_full_book_caveat=False):
    valid, problems = [], []
    for pair in pairs:
        a, b = pair.get("before"), pair.get("after")
        if not a or not b or a.get("status") not in ("ok", "ok_with_recovered_errors") or b.get("status") not in ("ok", "ok_with_recovered_errors"):
            problems.append(f"pair {pair['index']}: failed measurement")
            continue
        if not a.get("validation", {}).get("valid") or not b.get("validation", {}).get("valid"):
            problems.append(f"pair {pair['index']}: output validation missing/invalid")
            continue
        if not a.get("integrity_unchanged", True) or not b.get("integrity_unchanged", True):
            problems.append(f"pair {pair['index']}: input or executable changed during measurement")
            continue
        if not workload_is_comparable(a) or not workload_is_comparable(b):
            problems.append(f"pair {pair['index']}: missing or inconsistent page workload evidence")
            continue
        if expected_pages is not None and (len(workload(a)) != expected_pages or len(workload(b)) != expected_pages):
            problems.append(f"pair {pair['index']}: selected page count differs from expected {expected_pages}")
            continue
        if workload(a) != workload(b):
            problems.append(f"pair {pair['index']}: page workload differs")
            continue
        valid.append((a, b))
    metrics = {}
    for key in ("wall_s", "rss_peak_bytes", "job_commit_peak_bytes", "cpu_s"):
        av, bv, ratios = [], [], []
        for a, b in valid:
            x, y = resources(a)[key], resources(b)[key]
            if x is not None and y is not None and x != 0:
                av.append(x); bv.append(y); ratios.append(y / x)
        metrics[key] = dict(before=dist(av), after=dist(bv), paired_after_over_before=dist(ratios),
                            change_percent=(statistics.median(ratios) - 1) * 100 if ratios else None)
    gpu_ids = sorted({uuid for a, b in valid for run in (a, b) for uuid in resources(run)["gpu_memory_peak_mib"]})
    for uuid in gpu_ids:
        av, bv, ratios = [], [], []
        for a, b in valid:
            x = resources(a)["gpu_memory_peak_mib"].get(uuid)
            y = resources(b)["gpu_memory_peak_mib"].get(uuid)
            if x is not None and y is not None and x:
                av.append(x); bv.append(y); ratios.append(y / x)
        metrics["gpu_memory_mib:" + uuid] = dict(before=dist(av), after=dist(bv), paired_after_over_before=dist(ratios),
                                                   change_percent=(statistics.median(ratios)-1)*100 if ratios else None)
    stage_names = sorted({s["stage"] for a, b in valid for run in (a, b)
                          for s in run.get("analysis", {}).get("stages", []) if s["kind"] in ("operation", "request")})
    for stage in stage_names:
        vals = {side: [] for side in ("before", "after")}
        ratios = []
        for a, b in valid:
            pair_vals = []
            for side, run in (("before", a), ("after", b)):
                found = next((s["total_ms"] for s in run.get("analysis", {}).get("stages", [])
                              if s["stage"] == stage and s["kind"] in ("operation", "request")), None)
                if found is not None: vals[side].append(found); pair_vals.append(found)
            if len(pair_vals) == 2 and pair_vals[0]: ratios.append(pair_vals[1] / pair_vals[0])
        metrics["stage:" + stage] = dict(before=dist(vals["before"]), after=dist(vals["after"]),
                                         paired_after_over_before=dist(ratios),
                                         change_percent=(statistics.median(ratios)-1)*100 if ratios else None)
    return dict(verdict="inconclusive" if problems or len(valid) < 5 else "descriptive comparison",
                valid_pairs=len(valid), problems=problems, metrics=metrics,
                limitations=["有效配对少于 5 时证据不足。" if len(valid) < 5 else "已达到 5 个有效配对；结果仍是描述性比较。",
                             "小样本 p95 不稳定，且此报告不据此主张因果优化收益。"] +
                            (["已知全本漏检限制适用于此全本运行，不能解释为成功无漏译。"] if include_full_book_caveat else []))


def render(report):
    c = report["comparison"]
    lines = ["# 配对基准比较", "", f"判定：**{c['verdict']}**；有效配对 {c['valid_pairs']}/{report['settings']['pairs']}。", "",
             "失败测量和逐页工作量不一致的配对不进入性能统计；逐轮原始证据仍保留。", "",
             "| 指标 | Before 中位数（范围） | After 中位数（范围） | 配对比值 After/Before | 变化 | n |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in c["metrics"].items():
        def fmt(d): return "—" if d is None else f"{d['median']:.3f} ({d['minimum']:.3f}–{d['maximum']:.3f})"
        ratio = m["paired_after_over_before"]
        change = "—" if m["change_percent"] is None else f"{m['change_percent']:+.1f}%"
        lines.append(f"| {name} | {fmt(m['before'])} | {fmt(m['after'])} | {fmt(ratio)} | "
                     f"{change} | {ratio['n'] if ratio else 0} |")
    sample_note = ("有效配对少于 5：证据不足。" if c["valid_pairs"] < 5 else "有效配对达到 5；仍应谨慎解读小样本 p95。")
    lines += ["", sample_note + " 小样本 p95 不稳定，不作为稳定尾延迟结论。", "",
              "## 配对情况", ""]
    for p in report["pairs"]:
        lines.append(f"- Pair {p['index']}: order {p['order']}, before={p.get('before', {}).get('status', 'missing')}, after={p.get('after', {}).get('status', 'missing')}")
    lines += ["", "## 限制", "", *[f"- {x}" for x in c["limitations"]], ""]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="Selected page limit; 0 means all pages")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--before-arg", action="append", default=[])
    p.add_argument("--after-arg", action="append", default=[])
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    if args.pairs < 1 or args.limit < 0: p.error("--pairs must be >= 1 and --limit >= 0")
    src, before, after = (x.resolve(strict=True) for x in (args.input, args.before, args.after))
    out = args.output_dir.resolve()
    ids = {"input": digest(src), "before": digest(before), "after": digest(after)}
    out.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = out / "comparison.json"
    settings = dict(input=str(src), before=str(before), after=str(after), pairs=args.pairs, limit=args.limit,
                    before_args=args.before_arg, after_args=args.after_arg)
    expected_pages = selected_page_count(src, args.limit)
    if args.resume and manifest_path.exists():
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report["identities"] != ids or report["settings"] != settings:
            p.error("resume identity/settings mismatch; choose a new output directory")
    else:
        report = dict(schema=1, started_utc=datetime.now(timezone.utc).isoformat(), identities=ids,
                      settings=settings, pairs=[])
    for i in range(1, args.pairs + 1):
        existing = next((x for x in report["pairs"] if x["index"] == i and x.get("complete")), None)
        if existing:
            print(f"pair {i}/{args.pairs}: checkpoint exists; skipping", flush=True); continue
        order = ["before", "after"] if i % 2 else ["after", "before"]
        prior = next((x for x in report["pairs"] if x["index"] == i), {})
        row = prior | dict(index=i, order="".join("A" if x == "before" else "B" for x in order), complete=False)
        pair_dir = out / f"pair-{i:03d}"; pair_dir.mkdir(exist_ok=True)
        for side in order:
            if side in row:
                print(f"pair {i}/{args.pairs}: {side.upper()} checkpoint exists; skipping", flush=True)
                continue
            exe, extra = (before, args.before_arg) if side == "before" else (after, args.after_arg)
            run_dir = pair_dir / side
            command = [sys.executable, str(Path(__file__).with_name("bench.py")), str(src), "--exe", str(exe),
                       "--runs", "1", "--warmup", "0", "--limit", str(args.limit), "--output-dir", str(run_dir),
                       *engine_arg_tokens(extra)]
            if run_dir.exists():
                report_path = run_dir / "report.json"
                if not report_path.exists():
                    raise RuntimeError(f"existing bench directory has no report; stopping safely: {run_dir}")
                try:
                    row[side] = completed_result(json.loads(report_path.read_text(encoding="utf-8")))
                except ValueError as exc:
                    raise RuntimeError(f"{exc}: {run_dir}") from exc
                print(f"pair {i}/{args.pairs}: restored completed {side.upper()} report", flush=True)
                report["pairs"] = [x for x in report["pairs"] if x["index"] != i] + [row]
                report["pairs"].sort(key=lambda x: x["index"])
                manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            (pair_dir / f"{side}-command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"pair {i}/{args.pairs}: running {side.upper()} ({row['order']})", flush=True)
            proc = subprocess.run(command, capture_output=True, text=True)
            (pair_dir / f"{side}-console.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
            report_path = run_dir / "report.json"
            if report_path.exists():
                detail = json.loads(report_path.read_text(encoding="utf-8"))
                try:
                    row[side] = completed_result(detail)
                except ValueError as exc:
                    raise RuntimeError(f"{exc}: {run_dir}") from exc
            else:
                row[side] = dict(status="infrastructure_failure", issues=[f"bench.py exit {proc.returncode}"])
            report["pairs"] = [x for x in report["pairs"] if x["index"] != i] + [row]
            report["pairs"].sort(key=lambda x: x["index"])
            manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        row["complete"] = True
        report["pairs"] = [x for x in report["pairs"] if x["index"] != i] + [row]
        report["pairs"].sort(key=lambda x: x["index"])
        report["comparison"] = comparison(report["pairs"], expected_pages,
                                          include_full_book_caveat=(args.limit == 0))
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "comparison.md").write_text(render(report), encoding="utf-8")
        print(f"pair {i}/{args.pairs}: checkpoint saved; verdict {report['comparison']['verdict']}", flush=True)
    report["comparison"] = comparison(report["pairs"], expected_pages,
                                      include_full_book_caveat=(args.limit == 0))
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "comparison.md").write_text(render(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
