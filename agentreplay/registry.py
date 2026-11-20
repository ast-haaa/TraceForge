# agentreplay/registry.py
"""Agent registration system for AgentReplay (Phase 6a).

Allows agents (e.g. refund_agent, lg_refund_agent, custom agents) to register
their main execution runners, fork step runners, and replay patch targets.
Replaces hardcoded dispatch in fork.py and hardcoded patch lists in replay.py.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class UnknownAgentError(ValueError):
    """Raised when an unrecognised agent name is encountered."""

    pass


@dataclass
class AgentRegistration:
    """Registration record for an agent."""

    name: str
    runner: Callable[..., Any] | None = None
    runner_from_step: Callable[..., Any] | None = None
    patch_targets: list[Any] = field(default_factory=list)


_REGISTRY: dict[str, AgentRegistration] = {}


def register_agent(
    name: str,
    runner: Callable[..., Any] | None = None,
    runner_from_step: Callable[..., Any] | None = None,
    patch_targets: list[Any] | None = None,
) -> AgentRegistration:
    """Register an agent with its execution runners and replay patch targets.

    Args:
        name: Unique name of the agent (e.g. 'refund_agent', 'lg_refund_agent').
        runner: Main entrypoint function to execute the agent.
        runner_from_step: Function to execute the agent from a given step (for forking).
        patch_targets: List of import strings or (import_string, mock_fn) tuples to patch during replay.

    Returns:
        The created AgentRegistration instance.
    """
    reg = AgentRegistration(
        name=name,
        runner=runner,
        runner_from_step=runner_from_step,
        patch_targets=patch_targets or [],
    )
    _REGISTRY[name] = reg
    return reg


def _ensure_builtins_registered() -> None:
    """Lazy-load demo agents if not already registered."""
    if "refund_agent" not in _REGISTRY:
        try:
            import demo_agent.agent  # noqa: F401
        except Exception:
            pass
    if "lg_refund_agent" not in _REGISTRY:
        try:
            import demo_agent.lg_agent  # noqa: F401
        except Exception:
            pass


def get_agent(name: str) -> AgentRegistration:
    """Retrieve registration for an agent by name.

    Raises:
        UnknownAgentError: If the agent has not been registered.
    """
    if name not in _REGISTRY:
        _ensure_builtins_registered()

    if name not in _REGISTRY:
        raise UnknownAgentError(
            f"Unknown agent '{name}'. Please register it with "
            f"`agentreplay.register_agent(name='{name}', runner=..., runner_from_step=..., patch_targets=...)` "
            "before replaying or forking."
        )
    return _REGISTRY[name]


def get_all_registrations() -> dict[str, AgentRegistration]:
    """Retrieve all registered agents, ensuring builtins are loaded."""
    _ensure_builtins_registered()
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Clear all agent registrations (for testing)."""
    _REGISTRY.clear()
