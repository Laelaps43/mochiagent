from pathlib import Path
from unittest.mock import patch

from agent.common.skill.loader import SkillLoader


def test_missing_skill_does_not_use_print(tmp_path: Path):
    loader = SkillLoader(tmp_path)

    with patch("builtins.print") as mocked_print:
        skill = loader.load_skill("missing")

    assert skill is None
    mocked_print.assert_not_called()


def test_valid_skill_does_not_use_print(tmp_path: Path):
    skill_dir = tmp_path / "data-analysis"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: data-analysis\ndescription: test skill\n---\n\ncontent",
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)

    with patch("builtins.print") as mocked_print:
        skill = loader.load_skill("data-analysis")

    assert skill is not None
    assert skill.name == "data-analysis"
    mocked_print.assert_not_called()


def test_path_traversal_skill_name_is_rejected(tmp_path: Path):
    loader = SkillLoader(tmp_path)

    with patch("builtins.print") as mocked_print:
        skill = loader.load_skill("../../etc/passwd")

    assert skill is None
    mocked_print.assert_not_called()
