import unittest
from argparse import ArgumentParser

from arguments import ModelParams, _backfill_model_compatibility


class ModelArgumentCompatibilityTest(unittest.TestCase):
    def test_old_config_gets_new_defaults_without_overwriting_existing_values(self):
        old = {"num_sg": 6, "reflectance_detail_scale": 0.4, "reflectance_highfreq_target_ratio": 0.7}
        result = _backfill_model_compatibility(old)
        self.assertEqual(result["sg_lobes"], 6)
        self.assertEqual(result["reflectance_detail_scale"], 0.4)
        self.assertEqual(result["reflectance_highfreq_target_ratio"], 0.7)
        self.assertIn("reflectance_decoder_scale", result)
        self.assertIn("reflectance_contrast_kernel_size", result)

    def test_representation_parameters_are_extracted_from_cli(self):
        parser = ArgumentParser()
        model = ModelParams(parser)
        arguments = parser.parse_args(["--reflectance_detail_scale", "0.4", "--reflectance_decoder_scale", "0.2",
                                       "--reflectance_contrast_kernel_size", "7"])
        extracted = model.extract(arguments)
        self.assertEqual(extracted.reflectance_detail_scale, 0.4)
        self.assertEqual(extracted.reflectance_decoder_scale, 0.2)
        self.assertEqual(extracted.reflectance_contrast_kernel_size, 7)


if __name__ == "__main__":
    unittest.main()
