import unittest

from app.core.database import SessionLocal
from app.services.library_service import LibraryService


class LibraryServiceTests(unittest.TestCase):
    def test_recognition_model_options_are_deduped(self):
        db = SessionLocal()
        try:
            service = LibraryService(db)
            options = service.get_options()
        finally:
            db.close()

        recognition_keys = [item["key"] for item in options["recognition_models"]]
        self.assertIn("production/recognition/resnet50_anime_arcface_v2.pth", recognition_keys)
        self.assertIn("production/recognition/resnet50_anime_arcface_best_r4.pth", recognition_keys)
        self.assertIn("archive/recognition/resnet50_anime_arcface.pth", recognition_keys)
        self.assertNotIn("catalog/recognition/20260427_215348_recognition_arcface_r4_demo.pth", recognition_keys)


if __name__ == "__main__":
    unittest.main()
