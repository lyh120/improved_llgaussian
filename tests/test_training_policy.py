import unittest
from argparse import ArgumentParser
from types import SimpleNamespace

from arguments import ModelParams
from utils.training_policy import prepare_warmup_args, residual_stage_state, validate_training_args


class TrainingPolicyTest(unittest.TestCase):
    def test_residual_cli_flag_remains_enabled(self):
        parser = ArgumentParser()
        ModelParams(parser)
        args = parser.parse_args(["--use_residual", "--prune_ratio", "0.1"])
        validate_training_args(args)
        self.assertTrue(args.use_residual)
        self.assertFalse(args.use_dual_transient)
        self.assertEqual(args.prune_ratio, 0.1)

    def test_warmup_preserves_initial_retention_without_mutating_main_args(self):
        args = SimpleNamespace(iterations=6000, save_iterations=[4000, 6000],
                               use_residual=True, use_dual_transient=False, prune_ratio=0.1)
        warmup = prepare_warmup_args(args)
        self.assertEqual(warmup.prune_ratio, 0.1)
        self.assertTrue(warmup.use_residual)
        self.assertEqual(warmup.iterations, 2000)
        self.assertIn(2000, warmup.save_iterations)
        self.assertEqual(args.save_iterations, [4000, 6000])
        self.assertEqual(args.iterations, 6000)

    def test_single_residual_starts_immediately_in_main_and_not_warmup(self):
        self.assertEqual(residual_stage_state(True, False, "warmup", 1000, 3000, 2500), (False, 0.0))
        self.assertEqual(residual_stage_state(True, False, "warmuped", 1, 3000, 2500), (True, 1.0))
        self.assertEqual(residual_stage_state(True, False, "train", 1, 3000, 2500), (True, 1.0))
        self.assertEqual(residual_stage_state(False, False, "train", 4000, 3000, 2500), (False, 0.0))

    def test_dual_transient_keeps_its_existing_schedule(self):
        self.assertEqual(residual_stage_state(True, True, "train", 2999, 3000, 2500), (False, 0.0))
        active, weight = residual_stage_state(True, True, "train", 3000, 3000, 2500)
        self.assertTrue(active)
        self.assertAlmostEqual(weight, 1/2500)
        self.assertEqual(residual_stage_state(True, True, "train", 6000, 3000, 2500), (True, 1.0))

    def test_invalid_retention_and_branch_combinations_are_rejected(self):
        for ratio, residual, dual in ((0, True, False), (1.1, True, False), (0.1, False, True)):
            with self.assertRaises(ValueError):
                validate_training_args(SimpleNamespace(prune_ratio=ratio, use_residual=residual,
                                                        use_dual_transient=dual))


if __name__ == "__main__":
    unittest.main()
