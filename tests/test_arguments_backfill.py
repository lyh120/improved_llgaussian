import argparse
import unittest

from arguments import ModelParams, OptimizationParams, _backfill_model_compatibility


class BackfillDefaultsTest(unittest.TestCase):
    def test_backfill_defaults_match_param_groups(self):
        parser = argparse.ArgumentParser()
        model_params = ModelParams(parser)
        opt_params = OptimizationParams(parser)

        backfilled = _backfill_model_compatibility({})
        sources = {**vars(model_params), **vars(opt_params)}

        missing = sorted(key for key in backfilled if key not in sources)
        mismatched = sorted(
            key
            for key in backfilled
            if key in sources and sources[key] != backfilled[key]
        )
        self.assertEqual(missing, [], "backfill keys missing from param groups")
        self.assertEqual(mismatched, [], "backfill defaults diverging from param groups")


if __name__ == "__main__":
    unittest.main()
