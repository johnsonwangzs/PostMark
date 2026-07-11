from __future__ import annotations

import math
import unittest

from postmark.common import ConfigurationError, DuplicateIdError
from postmark.metrics import calibration_threshold, evaluate_paired_scores, roc_auc


class CalibrationMetricTests(unittest.TestCase):
    def test_threshold_uses_complete_tie_groups_and_finite_sentinel(self) -> None:
        threshold, fpr = calibration_threshold([0.0, 0.0, 0.0], target_fpr=0.01)
        self.assertTrue(math.isfinite(threshold))
        self.assertGreater(threshold, 0.0)
        self.assertEqual(fpr, 0.0)

        threshold, fpr = calibration_threshold(
            [0.1] * 99 + [0.9], target_fpr=0.01
        )
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.9)
        self.assertEqual(fpr, 0.01)

    def test_auc_handles_ties(self) -> None:
        self.assertEqual(roc_auc([0.0, 0.5], [0.5, 1.0]), 0.875)
        self.assertEqual(roc_auc([0.5], [0.5]), 0.5)

    def test_paired_metrics_are_reorder_and_seed_stable(self) -> None:
        arguments = {
            "sample_ids": ["b", "a", "c"],
            "negative_scores": [0.2, 0.1, 0.4],
            "positive_scores": [0.8, 0.6, 0.9],
            "calibration_ids": ["n2", "n1", "n3"],
            "calibration_negative_scores": [0.2, 0.1, 0.3],
            "detector_config_sha256": "detector",
            "bootstrap_samples": 100,
            "bootstrap_seed": 7,
        }
        first, first_config = evaluate_paired_scores(**arguments)
        reordered, reordered_config = evaluate_paired_scores(
            **{
                **arguments,
                "sample_ids": ["a", "c", "b"],
                "negative_scores": [0.1, 0.4, 0.2],
                "positive_scores": [0.6, 0.9, 0.8],
                "calibration_ids": ["n3", "n2", "n1"],
                "calibration_negative_scores": [0.3, 0.2, 0.1],
            }
        )
        self.assertEqual(first, reordered)
        self.assertEqual(first_config, reordered_config)
        self.assertEqual(first["roc_auc"], 1.0)
        self.assertEqual(first["metric_status"], "diagnostic_insufficient_negatives")

    def test_heldout_scores_cannot_change_frozen_threshold(self) -> None:
        common = {
            "sample_ids": ["a", "b"],
            "negative_scores": [0.1, 0.2],
            "calibration_ids": ["n1", "n2"],
            "calibration_negative_scores": [0.3, 0.4],
            "detector_config_sha256": "detector",
            "bootstrap_samples": 10,
        }
        first, _ = evaluate_paired_scores(
            **common, positive_scores=[0.5, 0.6]
        )
        second, _ = evaluate_paired_scores(
            **common, positive_scores=[0.0, 1.0]
        )
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["calibration_fpr"], second["calibration_fpr"])

    def test_split_overlap_duplicates_and_nonfinite_scores_are_rejected(self) -> None:
        base = {
            "sample_ids": ["a"],
            "negative_scores": [0.1],
            "positive_scores": [0.2],
            "calibration_ids": ["n"],
            "calibration_negative_scores": [0.1],
            "detector_config_sha256": "detector",
            "bootstrap_samples": 2,
        }
        with self.assertRaises(ConfigurationError):
            evaluate_paired_scores(**{**base, "calibration_ids": ["a"]})
        with self.assertRaises(DuplicateIdError):
            evaluate_paired_scores(
                **{
                    **base,
                    "sample_ids": ["a", "a"],
                    "negative_scores": [0.1, 0.1],
                    "positive_scores": [0.2, 0.2],
                }
            )
        with self.assertRaises(ConfigurationError):
            evaluate_paired_scores(**{**base, "positive_scores": [math.nan]})


if __name__ == "__main__":
    unittest.main()
