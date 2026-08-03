import json
import tempfile
import unittest
from pathlib import Path

from examples.supo_browsecomp.mast.train_data_filter.aggregate import aggregate


class AggregateTest(unittest.TestCase):
    def test_intersects_strict_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = {
                "run_name": "run",
                "points": [
                    {"name": "base", "step": "base"},
                    {"name": "iter04", "step": 4},
                    {"name": "iter09", "step": 9},
                ],
                "evaluation": {"expected_questions": 2, "samples_per_question": 8},
                "dataset": {"sha256": "dataset"},
            }
            (root / "config.json").write_text(json.dumps(config))
            for name, step, second_successes in (
                ("base", "base", 8),
                ("iter04", 4, 7),
                ("iter09", 9, 8),
            ):
                point = root / "points" / name
                point.mkdir(parents=True)
                (point / "_SUCCESS").write_text("{}")
                manifest = {
                    "model_name": "Qwen3.5-4B",
                    "dataset_sha256": "dataset",
                    "judge_model": "judge",
                    "sampling": {"samples_per_question": 8},
                    "load_verification": {"actual_step": step},
                }
                (point / "manifest.json").write_text(json.dumps(manifest))
                (point / "point_metrics.json").write_text(
                    json.dumps({"n_questions": 2, "samples_per_question": 8})
                )
                rows = [
                    {"query_id": "always", "question": "q1", "successes": 8},
                    {"query_id": "not-always", "question": "q2", "successes": second_successes},
                ]
                (point / "questions.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )

            summary = aggregate(root)
            self.assertEqual(summary["n_filter_candidates"], 1)
            candidates = [json.loads(line) for line in (root / "filter_candidates.jsonl").read_text().splitlines()]
            self.assertEqual([candidate["query_id"] for candidate in candidates], ["always"])


if __name__ == "__main__":
    unittest.main()
