import unittest

import torch

from utils.artifact_utils import (
    compute_artifact_diagnostics,
    compute_detail_diagnostics,
    summarize_artifact_diagnostics,
    summarize_detail_diagnostics,
)


class ArtifactDiagnosticsTest(unittest.TestCase):
    def test_black_artifact_and_coverage_overlap(self):
        prediction = torch.ones(3, 2, 2)
        prediction[:, 0, 0] = 0.0
        bright_gt = torch.ones(3, 2, 2)
        coverage = torch.ones(1, 2, 2)
        coverage[:, 0, 0] = 0.5

        result = compute_artifact_diagnostics(prediction, bright_gt, coverage)

        self.assertEqual(result["black_artifact_pixels"], 1)
        self.assertEqual(result["artifact_low_coverage_pixels"], 1)
        self.assertAlmostEqual(result["black_artifact_ratio"], 0.25)
        self.assertAlmostEqual(result["artifact_low_coverage_overlap_ratio"], 1.0)

        summary = summarize_artifact_diagnostics({"view.png": result})
        self.assertEqual(summary["black_artifact_pixels"], 1)
        self.assertAlmostEqual(summary["black_artifact_ratio"], 0.25)

    def test_rgb_coverage_does_not_duplicate_intersection(self):
        prediction = torch.ones(3, 2, 2)
        prediction[:, 0, 0] = 0.0
        bright_gt = torch.ones(3, 2, 2)
        coverage = torch.ones(3, 2, 2)
        coverage[:, 0, 0] = 0.5

        result = compute_artifact_diagnostics(prediction, bright_gt, coverage)

        self.assertEqual(result["black_artifact_pixels"], 1)
        self.assertEqual(result["artifact_low_coverage_pixels"], 1)
        self.assertLessEqual(result["artifact_low_coverage_overlap_ratio"], 1.0)

    def test_detail_recall_distinguishes_retained_and_missing_edge(self):
        bright_gt = torch.full((3, 8, 8), 0.2)
        bright_gt[:, :, 4:] = 0.8
        coverage = torch.ones(3, 8, 8)
        retained = compute_detail_diagnostics(bright_gt, bright_gt, coverage)
        missing = compute_detail_diagnostics(
            torch.full_like(bright_gt, 0.5),
            bright_gt,
            coverage,
        )

        self.assertGreater(retained["edge_recall"], 0.99)
        self.assertLess(missing["edge_recall"], retained["edge_recall"])
        summary = summarize_detail_diagnostics({"view.png": retained})
        self.assertAlmostEqual(summary["edge_recall"], retained["edge_recall"])


if __name__ == "__main__":
    unittest.main()
