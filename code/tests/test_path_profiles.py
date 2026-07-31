import os
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.util import (
    LOCAL_PROJECT_ROOT,
    SERVER_PROJECT_ROOT,
    load_config,
    rebase_project_path,
)


CODE_ROOT = Path(__file__).resolve().parents[1]


class PathProfileTests(unittest.TestCase):
    def test_local_profile_uses_current_checkout(self):
        cfg = load_config(str(CODE_ROOT / "configs" / "pku.yaml"), "local")
        self.assertEqual(cfg.project_root, str(LOCAL_PROJECT_ROOT))
        self.assertEqual(
            cfg.paths.base,
            str(LOCAL_PROJECT_ROOT / "data" / "dataset" / "pku" / "split"),
        )
        self.assertTrue(cfg.paths.train.inp_dir.startswith(str(LOCAL_PROJECT_ROOT)))

    def test_server_profile_uses_viplab_root(self):
        cfg = load_config(str(CODE_ROOT / "configs" / "cgl.yaml"), "server")
        self.assertEqual(cfg.project_root, str(SERVER_PROJECT_ROOT))
        self.assertEqual(
            cfg.paths.base,
            "/home/viplab/Aagha/intent_aware_layout_generation/data/dataset/cgl/split",
        )

    def test_environment_variable_selects_server(self):
        with patch.dict(os.environ, {"INTENTDIT_PATH_PROFILE": "server"}):
            cfg = load_config(str(CODE_ROOT / "configs" / "pku.yaml"))
        self.assertEqual(cfg.path_profile, "server")

    def test_old_checkpoint_path_rebases_to_profile(self):
        old = (
            "/media/erc/GPU/projects/aagha-ii/intent_aware_layout_generation/"
            "data/checkpoints/pku/model/Epoch400.pth"
        )
        self.assertEqual(
            rebase_project_path(old, "server"),
            "/home/viplab/Aagha/intent_aware_layout_generation/"
            "data/checkpoints/pku/model/Epoch400.pth",
        )
        self.assertEqual(
            rebase_project_path(old, "local"),
            str(LOCAL_PROJECT_ROOT / "data/checkpoints/pku/model/Epoch400.pth"),
        )


if __name__ == "__main__":
    unittest.main()
