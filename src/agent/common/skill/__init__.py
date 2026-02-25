"""
Skill system for agent capabilities enhancement.

Skills are markdown-based instruction files that provide specialized knowledge
and step-by-step guidance to agents. Each skill is loaded on-demand when an
agent explicitly registers it during setup.

Example:
    class AnalyticsAgent(BaseAgent):
        async def setup(self):
            self.register_skill("data-analysis")
            self.register_skill("sql-query")
"""

from .loader import Skill, SkillLoader

__all__ = ["Skill", "SkillLoader"]
