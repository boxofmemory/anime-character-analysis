import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

try:
    from app.services.character_service import CharacterService
    from app.services import character_service as character_service_module
except ModuleNotFoundError:
    CharacterService = None
    character_service_module = None


TMP_ROOT = Path(__file__).resolve().parents[1] / "_tmp"
TMP_ROOT.mkdir(parents=True, exist_ok=True)


@unittest.skipIf(CharacterService is None, "FastAPI dependencies are not installed in this Python environment")
class CharacterServiceTests(unittest.TestCase):
    def test_rename_stranger_creates_known_character(self):
        tmp = TMP_ROOT / "character_service_case"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        known = tmp / "known.json"
        stranger = tmp / "stranger.json"
        known.write_text("{}", encoding="utf-8")
        stranger.write_text(
            json.dumps({"Stranger_1": {"mean": [1.0, 0.0], "count": 1}}),
            encoding="utf-8",
        )

        with patch.object(
            type(character_service_module.settings),
            "known_db_path",
            new_callable=PropertyMock,
            return_value=known,
        ), patch.object(
            type(character_service_module.settings),
            "stranger_db_path",
            new_callable=PropertyMock,
            return_value=stranger,
        ):
            db = Mock()
            query = Mock()
            filtered = Mock()
            filtered.first.return_value = None
            query.filter.return_value = filtered
            db.query.return_value = query
            result = CharacterService(db).rename_stranger("Stranger_1", "Hoshino_Ai")

        self.assertTrue(result["created"])
        self.assertFalse(result["fused"])
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
