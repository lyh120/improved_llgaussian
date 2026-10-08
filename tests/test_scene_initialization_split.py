"""Keep held-out image colors out of data-dependent reflectance initialization."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scene import Scene


class SceneInitializationSplitTest(unittest.TestCase):
    def test_point_cloud_initialization_receives_training_views_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'sparse').mkdir()
            (root / 'points.ply').write_bytes(b'test point cloud')
            train_camera = SimpleNamespace(image_name='train')
            test_camera = SimpleNamespace(image_name='test')
            info = SimpleNamespace(
                train_cameras=[train_camera], test_cameras=[test_camera],
                nerf_normalization={'radius': 1.0}, point_cloud=object(),
                ply_path=str(root / 'points.ply'),
            )
            model = Mock()
            model.get_anchor = SimpleNamespace(shape=(1, 3))
            args = SimpleNamespace(model_path=str(root), source_path=str(root),
                                   images='images', eval=True, lod=0,
                                   prune_ratio=1.0, beta=1.0)
            with patch.dict('scene.sceneLoadTypeCallbacks', {'Colmap': Mock(return_value=info)}), \
                 patch('scene.cameraList_from_camInfos', side_effect=lambda cameras, *_: cameras), \
                 patch('scene.camera_to_JSON', return_value={}):
                Scene(args, model, depth_piror_model=None, shuffle=False)
            passed = model.create_from_pcd.call_args.kwargs['cameras']
            self.assertEqual(len(passed), 1)
            self.assertIs(passed[0], train_camera)
            self.assertIsNot(passed[0], test_camera)


if __name__ == '__main__':
    unittest.main()
