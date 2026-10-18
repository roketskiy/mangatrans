"""Run: python -m unittest discover -s scripts -p test_bench.py -v"""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

import bench

TEST_TEMP = bench.ROOT / ".cache" / "bench-tests"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


def image_bytes(color):
    buf = io.BytesIO()
    Image.new("RGB", (32, 48), color).save(buf, "PNG")
    return buf.getvalue()


class AnalysisTests(unittest.TestCase):
    def test_counter_events_are_reported_separately_from_time(self):
        trace = dict(schema=1, total_ns=1000, pages=[dict(index=0, name="p", width=2, height=3,
                    outcome="rendered", detected=1, ocr_accepted=1, translated=1, rendered=1)], events=[
            dict(kind="operation", stage="ocr_page", page=0, start_ns=100, duration_ns=200),
            dict(kind="counter", stage="ocr_boxes", page=0, count=3, start_ns=110, duration_ns=0)])
        result = bench.analyze_trace(trace)
        self.assertEqual(result["counters"]["ocr_boxes"]["total"], 3)
        self.assertEqual(result["page_operations"][0]["counters"]["ocr_boxes"], 3)
        self.assertEqual(result["stages"][0]["latency_ms"]["n"], 1)
        self.assertAlmostEqual(result["page_operations"][0]["measured_ms"], .0002)

    def test_counter_duration_must_be_zero(self):
        with self.assertRaises(ValueError):
            bench.analyze_trace(dict(schema=1, total_ns=10, pages=[], events=[
                dict(kind="counter", stage="n", count=2, start_ns=0, duration_ns=1)]))

    def test_nested_intervals_do_not_double_count(self):
        trace = dict(schema=1, total_ns=1000, pages=[], events=[
            dict(kind="operation", stage="a", start_ns=100, duration_ns=400),
            dict(kind="request", stage="http", start_ns=200, duration_ns=200),
            dict(kind="nested", stage="check", start_ns=600, duration_ns=100),
            dict(kind="operation", stage="b", start_ns=400, duration_ns=400)])
        for event in trace["events"]:
            event["page"] = -1
        result = bench.analyze_trace(trace)
        self.assertAlmostEqual(result["covered_ms"], .0007)
        self.assertAlmostEqual(result["uninstrumented_ms"], .0003)

    def test_failed_runs_do_not_enter_success_statistics(self):
        good = dict(warmup=False, status="ok", wall_s=2,
                    analysis=dict(pages=1, stages=[]))
        bad = good | dict(status="failed_or_degraded", wall_s=.001)
        warmup = good | dict(warmup=True, wall_s=100)
        result = bench.summarize([good, bad, warmup])
        self.assertEqual(result["successful"], 1)
        self.assertEqual(result["failed_or_degraded"], 1)
        self.assertEqual(result["wall_s"]["median"], 2)

    def test_validation_reads_pixels_and_checks_retained_pages(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            source, output = Path(tmp) / "input.cbz", Path(tmp) / "out.cbz"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("page10.png", image_bytes("white"))
                z.writestr("page2.png", image_bytes("white"))
            with zipfile.ZipFile(output, "w") as z:
                z.writestr("page2.png", image_bytes("black"))
            page = dict(name="page2.png", outcome="rendered", rendered=1, width=32, height=48)
            result = bench.validate_output(source, output, 1, dict(pages=[page]))
            self.assertEqual(result["changed_pages"], 1)
            page["outcome"] = "no_translation"
            with self.assertRaisesRegex(ValueError, "retained page"):
                bench.validate_output(source, output, 1, dict(pages=[page]))
            page["outcome"] = "rendered"
            with zipfile.ZipFile(output, "w") as z:
                z.writestr("page2.png", b"not an image")
            with self.assertRaises(Exception):
                bench.validate_output(source, output, 1, dict(pages=[page]))

    def test_reject_invalid_intervals(self):
        with self.assertRaises(ValueError):
            bench.analyze_trace(dict(schema=1, total_ns=10, pages=[], events=[
                dict(kind="operation", stage="bad", start_ns=0, duration_ns=11)]))

    def test_moved_input_is_an_integrity_failure(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            source = Path(tmp) / "input.cbz"
            source.write_bytes(b"original")
            expected = bench.identity(source)
            source.rename(Path(tmp) / "moved.cbz")
            result = bench.check_identity(source, expected)
            self.assertFalse(result["unchanged"])
            self.assertIn("FileNotFoundError", result["error"])

    def test_resource_alignment_drops_cross_boundary_windows(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            directory = Path(tmp)
            samples = [dict(elapsed_s=t, collection_ms=10, tree_rss_bytes=int(t*100), job=dict(cpu_s=t))
                       for t in (1.5, 2.5, 3.5, 4.5, 5.5)]
            (directory / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
            bench.dump(directory / "gpu.json", dict(samples=[]))
            result = dict(clock=dict(start_counter_ns=0, finish_counter_ns=10_000_000_000))
            trace = dict(clock=dict(qpc_ns=1_000_000_000, go_ns=0, uncertainty_ns=100),
                         events=[dict(kind="operation", stage="test", start_ns=1_000_000_000, duration_ns=3_000_000_000)])
            aligned = bench.align_resources(result, trace, directory)
            stage = aligned["stages"]["test"]
            self.assertTrue(aligned["available"])
            self.assertEqual(stage["rss_samples"], 3)
            self.assertEqual(stage["sampled_rss_peak_bytes"], 450)
            self.assertEqual(stage["cpu_s_inside_sample_windows"], 2)
            self.assertEqual(stage["cpu_sample_window_s"], 2)

    def test_recovered_retries_remain_in_success_summary_but_lost_ocr_does_not(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            directory = Path(tmp)
            source, output, path = directory / "in.cbz", directory / "out.cbz", directory / "trace.json"
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("p.png", image_bytes("white"))
            with zipfile.ZipFile(output, "w") as z:
                z.writestr("p.png", image_bytes("black"))
            page = dict(index=0, name="p.png", width=32, height=48, detected=1,
                        ocr_accepted=1, translated=1, rendered=1, outcome="rendered", ocr_failed=0)
            trace = dict(schema=1, total_ns=1000, pages=[page], events=[
                dict(kind="request", stage="llm_http", page=0, item=0,
                     start_ns=100, duration_ns=500, error="schema rejected; retried successfully")])
            base = dict(exit_code=0, timed_out=False, interrupted=False, residual_children=0, wall_s=1)
            path.write_text(json.dumps(trace), encoding="utf-8")
            good = bench.evaluate(base.copy(), source, output, 0, path)
            self.assertEqual(good["status"], "ok_with_recovered_errors")
            self.assertTrue(good["validation"]["valid"])
            good["warmup"] = False
            self.assertEqual(bench.summarize([good])["wall_s"]["n"], 1)
            page["ocr_failed"] = 1
            path.write_text(json.dumps(trace), encoding="utf-8")
            bad = bench.evaluate(base.copy(), source, output, 0, path)
            self.assertEqual(bad["status"], "failed_or_degraded")


@unittest.skipUnless(os.name == "nt", "Windows Job Object integration")
class ProcessTests(unittest.TestCase):
    def run_workload(self, command, directory, timeout=10):
        with patch("bench.shutil.which", return_value=None):
            return bench.run_process(command, directory, .05, timeout, 1)

    def test_exited_children_cpu_io_and_process_count_are_retained(self):
        with tempfile.TemporaryDirectory(prefix="bench test ", dir=TEST_TEMP) as tmp:
            directory = Path(tmp)
            child = directory / "child.py"
            child.write_text(
                "import time,sys\n"
                "data = bytearray(32*1024*1024)\n"
                "end = time.process_time() + .2\n"
                "while time.process_time() < end: sum(range(10000))\n"
                "with open(sys.argv[1], 'wb', buffering=0) as f: f.write(data)\n")
            root = directory / "root.py"
            root.write_text("import subprocess,sys,time\n"
                            "for i in range(2): subprocess.run([sys.executable,sys.argv[1],sys.argv[2]],check=True)\n"
                            "time.sleep(.15)\n")
            result = self.run_workload([sys.executable, str(root), str(child), str(directory / "data")], directory)
            self.assertEqual(result["exit_code"], 0)
            self.assertGreaterEqual(result["resources"]["job"]["total_processes"], 3)
            self.assertGreaterEqual(result["resources"]["job"]["cpu_s"], .35)
            self.assertGreaterEqual(result["resources"]["job"]["write_bytes"], 64 * 1024 * 1024)
            self.assertGreater(result["resources"]["job"]["peak_job_commit_bytes"], 32 * 1024 * 1024)
            self.assertEqual(result["residual_children"], 0)
            self.assertEqual(result["gpu"]["samples"], 0)

    def test_timeout_kills_whole_tree(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            directory = Path(tmp)
            root = directory / "root.py"
            pidfile = directory / "child.pid"
            root.write_text("import subprocess,sys,time,pathlib\n"
                            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
                            "pathlib.Path(sys.argv[1]).write_text(str(p.pid))\n"
                            "time.sleep(30)\n")
            result = self.run_workload([sys.executable, str(root), str(pidfile)], directory, timeout=1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertFalse(bench.psutil.pid_exists(int(pidfile.read_text())))

    def test_early_exit_is_recorded(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            result = self.run_workload([sys.executable, "-c", "raise SystemExit(7)"], Path(tmp))
            self.assertEqual(result["exit_code"], 7)
            self.assertGreater(result["wall_s"], 0)
            self.assertGreaterEqual(result["resources"]["job"]["total_processes"], 1)


if __name__ == "__main__":
    unittest.main()
