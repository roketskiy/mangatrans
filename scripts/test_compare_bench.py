"""Small deterministic tests for paired comparison decisions."""
import unittest

import compare_bench


def run(time, status="ok", work=1):
    page = dict(name="p", width=10, height=20, detected=work, ocr_accepted=work,
                ocr_failed=0, ocr_rejected=0, translated=work, rendered=work, outcome="rendered")
    return dict(status=status, wall_s=time, validation=dict(valid=True, pages=[dict(name="p")]),
                analysis=dict(pages=1, page_operations=[page], stages=[]), resources=dict(
                    sampled_tree_rss_peak_bytes=100, job=dict(peak_job_commit_bytes=200, cpu_s=1)), gpu={})


class CompareTests(unittest.TestCase):
    def test_only_work_matched_successful_pairs_contribute(self):
        pairs = [dict(index=1, before=run(10), after=run(8)),
                 dict(index=2, before=run(10), after=run(7, work=0)),
                 dict(index=3, before=run(10, "failed_or_degraded"), after=run(8))]
        result = compare_bench.comparison(pairs)
        self.assertEqual(result["valid_pairs"], 1)
        self.assertEqual(result["verdict"], "inconclusive")
        self.assertAlmostEqual(result["metrics"]["wall_s"]["change_percent"], -20)
        self.assertEqual(len(result["problems"]), 2)

    def test_fewer_than_five_pairs_is_inconclusive(self):
        self.assertEqual(compare_bench.comparison([dict(index=1, before=run(10), after=run(8))])["verdict"],
                         "inconclusive")

    def test_five_identical_workload_pairs_allow_descriptive_report(self):
        pairs = [dict(index=i, before=run(10), after=run(8)) for i in range(1, 6)]
        self.assertEqual(compare_bench.comparison(pairs)["verdict"], "descriptive comparison")

    def test_engine_arguments_keep_leading_dash_in_one_token(self):
        self.assertEqual(compare_bench.engine_arg_tokens(["-ocr-workers", "4"]),
                         ["--engine-arg=-ocr-workers", "--engine-arg=4"])

    def test_incomplete_report_is_rejected_and_empty_runs_do_not_index(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare_bench.completed_result({"runs": []})
        self.assertEqual(compare_bench.completed_result({"runs": [run(1)], "integrity": {},
                                                         "input_unchanged": True,
                                                         "executable_unchanged": True})["status"], "ok")

    def test_workload_must_be_nonempty_and_match_all_page_evidence(self):
        empty = dict(analysis=dict(pages=0, page_operations=[]), validation=dict(pages=[]))
        self.assertFalse(compare_bench.workload_is_comparable(empty))
        missing_validation = run(1)
        missing_validation["validation"]["pages"] = []
        self.assertFalse(compare_bench.workload_is_comparable(missing_validation))

    def test_nullable_gpu_memory_is_safe(self):
        row = run(1)
        row["gpu"] = dict(devices={"gpu0": dict(used_mib=None)})
        self.assertIsNone(compare_bench.resources(row)["gpu_memory_peak_mib"]["gpu0"])


if __name__ == "__main__":
    unittest.main()
