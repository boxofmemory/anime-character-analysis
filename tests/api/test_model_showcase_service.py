import unittest
from pathlib import Path

from app.core.settings import settings
from app.services.model_showcase_service import ModelShowcaseService


class ModelShowcaseServiceTests(unittest.TestCase):
    def test_showcase_returns_cards_and_experiment_groups(self):
        showcase = ModelShowcaseService().get_showcase()

        self.assertIn("cards", showcase)
        self.assertIn("experiment_groups", showcase)
        self.assertGreaterEqual(len(showcase["cards"]), 4)
        self.assertTrue(any(card["key"] == "yolo" for card in showcase["cards"]))
        self.assertTrue(any(group["task_type"] == "yolo" for group in showcase["experiment_groups"]))
        yolo_card = next(card for card in showcase["cards"] if card["key"] == "yolo")
        self.assertGreaterEqual(len(yolo_card["charts"]), 2)
        self.assertGreaterEqual(len(yolo_card["visual_groups"]), 1)

    def test_showcase_uses_selected_model_to_bind_experiment(self):
        selected_path = settings.catalog_recognition_dir / "20260427_215348_recognition_arcface_r4_demo.pth"
        showcase = ModelShowcaseService().get_showcase(
            recognition_model_path=str(selected_path),
        )

        recognition_card = next(card for card in showcase["cards"] if card["key"] == "recognition")
        recognition_group = next(group for group in showcase["experiment_groups"] if group["task_type"] == "recognition")

        self.assertEqual(recognition_card["artifact_name"], Path(selected_path).name)
        self.assertEqual(
            recognition_card["selected_experiment_id"],
            "20260427_215348_recognition_arcface_r4_demo",
        )
        self.assertGreaterEqual(len(recognition_card["charts"]), 3)
        self.assertTrue(any(experiment["selected"] for experiment in recognition_group["experiments"]))


if __name__ == "__main__":
    unittest.main()
