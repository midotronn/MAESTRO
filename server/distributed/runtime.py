"""Feature-gate helpers for staged distributed-role rollout."""

from __future__ import annotations

import os


def capability_enabled(capability: str) -> bool:
    enabled = os.environ.get("AGENTLODGE_DISTRIBUTED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False
    configured = os.environ.get(
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
        "",
    ).strip()
    if not configured:
        return True
    allowed = {
        item.strip().lower()
        for item in configured.split(",")
        if item.strip()
    }
    normalized = capability.strip().lower()
    role = normalized.split(".", 1)[0]
    return normalized in allowed or role in allowed


def distributed_transport(capability: str) -> str:
    configured = os.environ.get(
        "AGENTLODGE_DISTRIBUTED_TRANSPORT",
        "filesystem",
    ).strip().lower()
    aliases = {"file": "filesystem", "shared": "filesystem"}
    transport = aliases.get(configured, configured)
    if transport not in {"filesystem", "http"}:
        raise ValueError(f"unsupported distributed transport: {configured!r}")
    if transport == "http" and capability != "render.frames":
        raise ValueError(
            "HTTP distributed transport is currently scoped to render.frames"
        )
    return transport
