import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from app.core.settings import settings
from app.services.training_service import TrainingService


TMP_ROOT = Path(__file__).resolve().parents[1] / "_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)


class TrainingServiceTests(unittest.TestCase):
    def test_get_options_exposes_three_training_tasks(self):
        service = TrainingService()

        options = service.get_options()

        task_types = [task.task_type for task in options.tasks]
        self.assertEqual(task_types, ["yolo", "recognition", "tracker"])
        tracker_task = next(task for task in options.tasks if task.task_type == "tracker")
        self.assertGreaterEqual(len(tracker_task.primary_resource_options), 1)

    def test_list_jobs_reads_existing_experiment_directory(self):
        root = TMP_ROOT / "training_service_case"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        experiments_dir = root / "experiments"
        manifest_path = root / "training_production.json"
        manifest_path.write_text("{}", encoding="utf-8")

        job_dir = experiments_dir / "detection" / "20260418_120000_yolo_demo"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_id": job_dir.name,
                    "last_row": {
                        "metrics/precision(B)": 0.82,
                        "metrics/recall(B)": 0.76,
                        "metrics/mAP50(B)": 0.71,
                    },
                }
            ),
            encoding="utf-8",
        )
        (job_dir / "config.snapshot.yaml").write_text(
            f"dataset_yaml: {settings.training_root_dir / 'datasets' / 'yolo' / 'anime-face' / 'data.yaml'}\n",
            encoding="utf-8",
        )

        with patch.object(
            type(settings),
            "training_experiments_dir",
            new_callable=PropertyMock,
            return_value=experiments_dir,
        ), patch.object(
            type(settings),
            "training_production_manifest_path",
            new_callable=PropertyMock,
            return_value=manifest_path,
        ):
            jobs = TrainingService().list_jobs()

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].task_type, "yolo")
        self.assertEqual(jobs[0].job_id, job_dir.name)
        self.assertEqual(jobs[0].status, "completed")
        shutil.rmtree(root, ignore_errors=True)

    def test_catalog_models_are_exposed_as_primary_options(self):
        root = TMP_ROOT / "training_service_catalog_case"
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        catalog_recognition = root / "catalog" / "recognition"
        catalog_detection = root / "catalog" / "detection"
        catalog_tracking = root / "catalog" / "tracking"
        for path in (catalog_recognition, catalog_detection, catalog_tracking):
            path.mkdir(parents=True, exist_ok=True)

        (catalog_recognition / "imported_r4.pth").write_text("stub", encoding="utf-8")
        (catalog_detection / "imported_yolo.pt").write_text("stub", encoding="utf-8")
        (catalog_tracking / "imported_tracker.yaml").write_text("tracker: true\n", encoding="utf-8")

        with patch.object(
            type(settings),
            "catalog_recognition_dir",
            new_callable=PropertyMock,
            return_value=catalog_recognition,
        ), patch.object(
            type(settings),
            "catalog_detection_dir",
            new_callable=PropertyMock,
            return_value=catalog_detection,
        ), patch.object(
            type(settings),
            "catalog_tracking_dir",
            new_callable=PropertyMock,
            return_value=catalog_tracking,
        ):
            options = TrainingService().get_options()

        recognition_task = next(task for task in options.tasks if task.task_type == "recognition")
        detection_task = next(task for task in options.tasks if task.task_type == "yolo")
        tracker_task = next(task for task in options.tasks if task.task_type == "tracker")

        self.assertTrue(any(item.path.endswith("imported_r4.pth") for item in recognition_task.primary_resource_options))
        self.assertTrue(any(item.path.endswith("imported_yolo.pt") for item in detection_task.primary_resource_options))
        self.assertTrue(any(item.path.endswith("imported_tracker.yaml") for item in tracker_task.primary_resource_options))
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
