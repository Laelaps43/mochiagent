from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agent.common.skill.loader import Skill, SkillLoader


class _TestSkillLoader(SkillLoader):
    def extract_first_paragraph(self, content: str) -> str:
        return self._extract_first_paragraph(content)


def _write_skill_file(skill_root: Path, name: str, frontmatter_block: str, body: str) -> Path:
    skill_file = skill_root / name / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    _ = skill_file.write_text(frontmatter_block + body, encoding="utf-8")
    return skill_file


def test_load_skill_rejects_path_traversal(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    outside_skill = tmp_path / "evil" / "SKILL.md"
    outside_skill.parent.mkdir()
    _ = outside_skill.write_text("---\nname: evil\n---\nOutside skill", encoding="utf-8")

    loader = SkillLoader(skill_root)

    assert loader.load_skill("../evil") is None


def test_load_skill_returns_none_when_file_missing(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path)

    assert loader.load_skill("missing-skill") is None


def test_load_skill_returns_skill_when_file_exists(tmp_path: Path) -> None:
    skill_file = _write_skill_file(
        tmp_path,
        "my-skill",
        "---\nname: my-skill\ndescription: Does something useful\n---\n",
        "# My Skill\n\nThis is the skill content.",
    )
    loader = SkillLoader(tmp_path)

    skill = loader.load_skill("my-skill")

    assert skill is not None
    assert skill.name == "my-skill"
    assert skill.description == "Does something useful"
    assert skill.location == skill_file


def test_load_skill_returns_none_when_frontmatter_load_fails(tmp_path: Path) -> None:
    _ = _write_skill_file(tmp_path, "broken", "---\nname: broken\n---\n", "Broken content")
    loader = SkillLoader(tmp_path)

    with patch("agent.common.skill.loader.frontmatter.load", side_effect=RuntimeError("boom")):
        skill = loader.load_skill("broken")

    assert skill is None


def test_load_skill_uses_first_paragraph_when_description_missing(tmp_path: Path) -> None:
    _ = _write_skill_file(
        tmp_path,
        "fallback-skill",
        "---\nname: fallback-skill\n---\n",
        "# Heading\n\nFirst useful paragraph.\n\nSecond paragraph.",
    )
    loader = SkillLoader(tmp_path)

    skill = loader.load_skill("fallback-skill")

    assert skill is not None
    assert skill.description == "First useful paragraph."


def test_extract_first_paragraph_returns_first_non_heading_line(tmp_path: Path) -> None:
    loader = _TestSkillLoader(tmp_path)

    paragraph = loader.extract_first_paragraph("# Heading\n\nFirst paragraph\n\nSecond paragraph")

    assert paragraph == "First paragraph"


def test_extract_first_paragraph_returns_default_when_only_headings(tmp_path: Path) -> None:
    loader = _TestSkillLoader(tmp_path)

    paragraph = loader.extract_first_paragraph("# Heading\n\n## Subheading\n\n### Nested")

    assert paragraph == "No description available"


def test_skill_render_returns_content_without_context(tmp_path: Path) -> None:
    skill_file = _write_skill_file(
        tmp_path,
        "render-skill",
        "---\nname: render-skill\ndescription: Render skill\n---\n",
        "# My Skill\n\nUse $ARGUMENTS for dynamic input.",
    )
    skill = Skill(name="render-skill", description="Render skill", location=skill_file)

    rendered = skill.render()

    assert rendered == "# My Skill\n\nUse $ARGUMENTS for dynamic input."


def test_skill_render_replaces_arguments_when_context_present(tmp_path: Path) -> None:
    skill_file = _write_skill_file(
        tmp_path,
        "context-skill",
        "---\nname: context-skill\ndescription: Context skill\n---\n",
        "Input: $ARGUMENTS",
    )
    skill = Skill(name="context-skill", description="Context skill", location=skill_file)

    rendered = skill.render("structured input")

    assert rendered == "Input: structured input"


def test_skill_render_keeps_content_when_context_has_no_placeholder(tmp_path: Path) -> None:
    skill_file = _write_skill_file(
        tmp_path,
        "plain-skill",
        "---\nname: plain-skill\ndescription: Plain skill\n---\n",
        "Static content only",
    )
    skill = Skill(name="plain-skill", description="Plain skill", location=skill_file)

    rendered = skill.render("ignored context")

    assert rendered == "Static content only"
