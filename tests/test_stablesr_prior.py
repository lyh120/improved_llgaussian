import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.precompute_stablesr_prior import (
    _save_gained_input,
    _select_training_images,
)

ROOT = Path(__file__).resolve().parents[1]


class StableSRPriorTest(unittest.TestCase):
    def test_explicit_input_gain_is_applied_and_clamped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "output.png"
            image = Image.new("RGB", (2, 1))
            image.putdata([(10, 20, 30), (200, 130, 80)])
            image.save(source)

            size = _save_gained_input(source, output, gain=2.0)

            self.assertEqual(size, (2, 1))
            with Image.open(output) as gained:
                self.assertEqual(list(gained.getdata()), [(20, 40, 60), (255, 255, 160)])

    def test_eval_split_uses_training_views_only(self):
        images = {str(index): Path(f"{index}.png") for index in range(10)}
        selected = _select_training_images(images, evaluate=True, lod=0, llffhold=8)
        self.assertEqual(set(selected), {"1", "2", "3", "4", "5", "6", "7", "9"})

    def test_cli_wraps_backend_output_in_strict_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "scene" / "images"
            image_root.mkdir(parents=True)
            source = Image.new("RGB", (3, 2), color=(40, 20, 10))
            source.save(image_root / "view.png")

            stablesr_root = root / "StableSR"
            scripts_root = stablesr_root / "scripts"
            scripts_root.mkdir(parents=True)
            ldm_root = stablesr_root / "ldm"
            ldm_root.mkdir()
            (ldm_root / "__init__.py").write_text(
                "STABLESR_ROOT_IMPORT_OK = True\n",
                encoding="utf-8",
            )
            (scripts_root / "util_image.py").write_text(
                "STABLESR_SCRIPT_IMPORT_OK = True\n",
                encoding="utf-8",
            )
            fake_inference = scripts_root / "fake_inference.py"
            fake_inference.write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "from PIL import Image\n"
                "from ldm import STABLESR_ROOT_IMPORT_OK\n"
                "from util_image import STABLESR_SCRIPT_IMPORT_OK\n"
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale\n"
                "assert STABLESR_ROOT_IMPORT_OK\n"
                "assert STABLESR_SCRIPT_IMPORT_OK\n"
                "assert callable(rgb_to_grayscale)\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--init-img', required=True)\n"
                "p.add_argument('--outdir', required=True)\n"
                "a,_=p.parse_known_args()\n"
                "for src in Path(a.init_img).glob('*.png'):\n"
                "    Image.open(src).save(Path(a.outdir) / src.name)\n",
                encoding="utf-8",
            )
            config = root / "config.yaml"
            checkpoint = root / "stable.ckpt"
            vqgan = root / "vqgan.ckpt"
            for path in (config, checkpoint, vqgan):
                path.touch()
            output = root / "scene" / "diffusion_prior_2"

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "precompute_stablesr_prior.py"),
                    "--source_path",
                    str(root / "scene"),
                    "--output",
                    str(output),
                    "--stablesr_root",
                    str(stablesr_root),
                    "--inference_script",
                    str(fake_inference),
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(checkpoint),
                    "--vqgan_checkpoint",
                    str(vqgan),
                    "--input_gain",
                    "2",
                ],
                check=True,
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_format_version"], 2)
            self.assertEqual(manifest["prior_type"], "enhancement_rgb")
            self.assertEqual(manifest["entries"][0]["image_name"], "view")
            with Image.open(output / manifest["entries"][0]["file"]) as enhanced:
                self.assertEqual(enhanced.size, (3, 2))
                self.assertEqual(enhanced.getpixel((0, 0)), (80, 40, 20))


if __name__ == "__main__":
    unittest.main()
