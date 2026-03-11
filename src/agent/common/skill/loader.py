"""
Skill loader for on-demand skill loading from filesystem.

This module provides functionality to load individual skills from a directory
structure where each skill is defined in a SKILL.md file with YAML frontmatter.
"""

from pathlib import Path
from typing import Optional

import frontmatter
from loguru import logger


class Skill:
    """
    Represents a loaded skill with its metadata and content.

    A skill provides specialized knowledge and instructions that guide an agent's
    behavior for specific tasks. Skills are defined as Markdown files with YAML
    frontmatter containing metadata.
    """

    __slots__ = ("name", "description", "content", "location")

    def __init__(self, name: str, description: str, content: str, location: Path) -> None:
        self.name = name
        self.description = description
        self.content = content
        self.location = location

    def render(self, context: str = "") -> str:
        content = self.content
        if context and "$ARGUMENTS" in content:
            content = content.replace("$ARGUMENTS", context)
        return content


class SkillLoader:
    """
    On-demand loader for skills from a directory structure.

    Skills are organized in subdirectories, each containing a SKILL.md file:

        skills/
        ├── data-analysis/
        │   └── SKILL.md
        ├── sql-query/
        │   └── SKILL.md
        └── chart-design/
            └── SKILL.md
    """

    def __init__(self, skill_directory: Path):
        """
        Initialize the skill loader.

        Args:
            skill_directory: Root directory containing skill subdirectories
        """
        self.skill_directory = skill_directory

    def load_skill(self, name: str) -> Optional[Skill]:
        """
        Load a single skill by name.

        This method loads a skill on-demand from the filesystem. It looks for
        a SKILL.md file in a subdirectory matching the provided name.

        Args:
            name: Skill name (must match a subdirectory name)

        Returns:
            Skill object if found and successfully parsed, None otherwise
        """
        # Guard against path traversal in skill name.
        root = self.skill_directory.resolve(strict=False)
        skill_dir = (root / name).resolve(strict=False)
        try:
            skill_dir.relative_to(root)
        except ValueError:
            logger.error("Invalid skill name '{}' (outside skill directory '{}')", name, root)
            return None

        skill_file = skill_dir / "SKILL.md"

        if not skill_file.exists():
            logger.warning("Skill file not found: {}", skill_file)
            return None

        try:
            post = frontmatter.load(skill_file)
            skill_name = post.get("name") or name

            description = post.get("description")
            if not description:
                description = self._extract_first_paragraph(post.content)

            skill = Skill(
                name=skill_name,
                description=description,
                content=post.content,
                location=skill_file,
            )

            logger.info("Loaded skill: {}", skill_name)
            return skill

        except Exception as e:
            logger.error("Failed to load skill '{}': {}", name, e)
            return None

    def _extract_first_paragraph(self, content: str) -> str:
        """
        Extract the first non-empty, non-heading paragraph from markdown content.

        This is used as a fallback when no description is provided in frontmatter.

        Args:
            content: Markdown content

        Returns:
            First paragraph text, or a default message if none found
        """
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            # Skip empty lines and markdown headings
            if line and not line.startswith("#"):
                return line

        return "No description available"
