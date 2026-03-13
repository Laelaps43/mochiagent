from __future__ import annotations

from pathlib import Path

from agent.common.skill.loader import Skill
from agent.common.tools.skill_tool import SkillTool


def _make_skill(name: str, description: str, content: str, tmp_path: Path) -> Skill:
    skill_file = tmp_path / name / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    _ = skill_file.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{content}",
        encoding="utf-8",
    )
    return Skill(name=name, description=description, location=skill_file)


def test_skill_tool_name_returns_skill() -> None:
    tool = SkillTool({})

    assert tool.name == "skill"


def test_skill_tool_description_returns_empty_message_when_no_skills() -> None:
    tool = SkillTool({})

    assert tool.description == "No skills available."


def test_skill_tool_description_lists_skills_as_xml(tmp_path: Path) -> None:
    alpha = _make_skill("alpha", "Alpha description", "Alpha content", tmp_path)
    beta = _make_skill("beta", "Beta description", "Beta content", tmp_path)
    tool = SkillTool({"alpha": alpha, "beta": beta})

    description = tool.description

    assert "<available_skills>" in description
    assert "<name>alpha</name>" in description
    assert "<description>Alpha description</description>" in description
    assert "<name>beta</name>" in description
    assert description.endswith("</available_skills>")


def test_skill_tool_parameters_schema_requires_name() -> None:
    tool = SkillTool({})

    schema = tool.parameters_schema

    assert schema["type"] == "object"
    assert schema["required"] == ["name"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["name"] == {
        "type": "string",
        "description": "The skill identifier from available_skills",
    }


async def test_skill_tool_execute_returns_error_for_unknown_skill(tmp_path: Path) -> None:
    alpha = _make_skill("alpha", "Alpha description", "Alpha content", tmp_path)
    beta = _make_skill("beta", "Beta description", "Beta content", tmp_path)
    tool = SkillTool({"alpha": alpha, "beta": beta})

    result = await tool.execute(name="missing")

    assert result == "Error: Skill 'missing' not found. Available skills: alpha, beta"


async def test_skill_tool_execute_returns_formatted_content_without_context(tmp_path: Path) -> None:
    skill = _make_skill("alpha", "Alpha description", "Plain content", tmp_path)
    tool = SkillTool({"alpha": skill})

    result = await tool.execute(name="alpha")

    assert result == f"## Skill: alpha\n\n**Location**: {skill.location.parent}\n\nPlain content"


async def test_skill_tool_execute_passes_context_to_render(tmp_path: Path) -> None:
    skill = _make_skill("alpha", "Alpha description", "Use $ARGUMENTS now", tmp_path)
    tool = SkillTool({"alpha": skill})

    result = await tool.execute(name="alpha", context="focused input")

    assert (
        result
        == f"## Skill: alpha\n\n**Location**: {skill.location.parent}\n\nUse focused input now"
    )


async def test_skill_tool_execute_reports_none_when_registry_empty() -> None:
    tool = SkillTool({})

    result = await tool.execute(name="missing")

    assert result == "Error: Skill 'missing' not found. Available skills: none"
