import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mm_stack.config import StackConfig


class ConfigPathTests(unittest.TestCase):
    def test_env_overrides_repo_and_vault_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            vault_root = root / "vault"
            repo_root.mkdir()
            vault_root.mkdir()

            with patch.dict(
                os.environ,
                {
                    "SMART_STACK_ROOT": str(repo_root),
                    "SMART_STACK_VAULT_ROOT": str(vault_root),
                },
                clear=False,
            ):
                cfg = StackConfig()

            self.assertEqual(cfg.stack_root, repo_root)
            self.assertEqual(cfg.vault_root, vault_root)
            self.assertEqual(cfg.sqlite_path, vault_root / "smart_stack.db")
            self.assertEqual(cfg.lancedb_path, vault_root / "vectors.lance")
            self.assertEqual(cfg.inbox_dir, repo_root / "inbox")

    def test_defaults_fall_back_to_application_support_when_legacy_root_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_home = Path(tmp)
            expected_vault = fake_home / "Library" / "Application Support" / "SmartStack"

            with patch.dict(os.environ, {}, clear=True), patch("mm_stack.config.Path.home", return_value=fake_home):
                cfg = StackConfig()

            self.assertEqual(cfg.vault_root, expected_vault)
            self.assertEqual(cfg.media_dir, expected_vault / "Media")


if __name__ == "__main__":
    unittest.main()
