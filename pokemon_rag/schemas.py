"""Structured, descriptive goals; completion conditions are never executed."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    goal_type: Literal[
        "REACH_LOCATION", "COMPLETE_EVENT", "LEVEL_UP", "HEAL", "CATCH_POKEMON",
        "DEFEAT_TRAINER", "INSUFFICIENT_CONTEXT",
    ]
    target: str
    target_value: int | float | None = None
    reason: str
    completion_condition: str
    evidence: list[str]
