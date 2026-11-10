import unittest
from pathlib import Path

from app.core.settings import settings


class SettingsTests(unittest.TestCase):
    def test_runtime_paths_are_relative_to_project(self):
        self.assertTrue((settings.project_root / "apps").exists())
        self.assertTrue(isinstance(settings.known_db_path, Path))
        self.assertTrue(str(settings.known_db_path).endswith("known_database.json"))
        self.assertIn("models\\production\\detection", str(settings.detection_model_path))


if __name__ == "__main__":
    unittest.main()
