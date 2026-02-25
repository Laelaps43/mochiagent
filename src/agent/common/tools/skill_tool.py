"""
Skill tool implementation for exposing skills to LLM.

This module provides a unified Tool that allows the LLM to load and access
skills that have been registered by an agent. The tool dynamically generates
its description based on available skills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ...core.tools import Tool

if TYPE_CHECKING:
    from ..skill.loader import Skill


class SkillTool(Tool):
    """
    Unified tool for loading skills into context.

    This tool exposes all skills registered by an agent to the LLM. The tool's
    description dynamically lists all available skills with their names and
    descriptions, allowing the LLM to choose which skill to load based on the
    current task.

    When executed, the tool returns the full markdown content of the requested
    skill, which the LLM can then use to guide its work.

    Design:
        - Single tool named "skill" (not one tool per skill)
        - Description lists all available skills in XML format
        - Returns skill content as tool execution result
        - Supports optional context parameter for skill customization
    """

    def __init__(self, skills: Dict[str, Skill]):
        """
        Initialize the skill tool with available skills.

        Args:
            skills: Dictionary mapping skill names to Skill objects.
                    These are the skills registered by the agent.
        """
        self._skills = skills

    @property
    def name(self) -> str:
        return "skill"

    @property
    def description(self) -> str:
        if not self._skills:
            return "No skills available."

        desc_lines = [
            "Load a skill to get detailed instructions for a specific task.",
            "Skills provide specialized knowledge and step-by-step guidance.",
            "Use this when a task matches an available skill's description.",
            "Only the skills listed here are available:",
            "<available_skills>",
        ]

        for skill_name, skill in self._skills.items():
            desc_lines.extend(
                [
                    "  <skill>",
                    f"    <name>{skill_name}</name>",
                    f"    <description>{skill.description}</description>",
                    "  </skill>",
                ]
            )

        desc_lines.append("</available_skills>")
        return "\n".join(desc_lines)

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill identifier from available_skills",
                },
                "context": {
                    "type": "string",
                    "description": "Additional context for applying this skill (optional)",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, context: str = "", **kwargs) -> str:
        if name not in self._skills:
            available = ", ".join(self._skills.keys())
            return f"Error: Skill '{name}' not found. Available skills: {available or 'none'}"

        skill = self._skills[name]
        content = skill.render(context)

        output = [
            f"## Skill: {skill.name}",
            "",
            f"**Location**: {skill.location.parent}",
            "",
            content.strip(),
        ]

        return "\n".join(output)
