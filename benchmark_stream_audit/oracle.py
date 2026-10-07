"""Reusable bounded completion and partial-output oracle."""

from __future__ import annotations

from typing import Any
import json
from pathlib import Path


def classify_events(events: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> str:
    """Classify one normalized fixture event sequence."""

    if policy is None:
        path = Path(__file__).resolve().parents[1] / "policies/completion_oracle.json"
        policy = json.loads(path.read_text(encoding="utf-8"))
    rules = policy["classification_precedence"]
    for rule in rules:
        match = rule["match"]
        unknown = set(match) - {"event_type", "status_gte", "reason_in", "without_prior_terminal"}
        if unknown:
            raise ValueError(f"unsupported oracle predicates: {sorted(unknown)}")
    for rule in sorted(rules, key=lambda item: item["priority"]):
        match = rule["match"]
        for index, event in enumerate(events):
            if event.get("type") != match["event_type"]:
                continue
            if "status_gte" in match and event.get("status", 0) < match["status_gte"]:
                continue
            if "reason_in" in match and event.get("reason") not in match["reason_in"]:
                continue
            if match.get("without_prior_terminal") and any(
                item.get("type") in {"sse_finish", "sse_error"} for item in events[:index]
            ):
                continue
            return rule["outcome"]
    raise ValueError("event sequence has no recognized outcome")


def classify_output(events: list[dict[str, Any]], outcome: str) -> str:
    """Classify the text associated with an oracle outcome."""

    has_text = any(event.get("type") == "sse_content" and event.get("text") for event in events)
    if not has_text:
        return "no_text_output"
    return "complete_output" if outcome == "completed" else "partial_output"
