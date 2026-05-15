from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from lang2recover.sim.maniskill_pushcube import PushCubeState


class RecoveryStatus(str, Enum):
    NORMAL = "normal"
    NEEDS_RECOVERY = "needs_recovery"
    RECOVERED = "recovered"
    TASK_LIKELY_DONE = "task_likely_done"


@dataclass(frozen=True)
class RecoveryDecision:
    status: RecoveryStatus
    reason: str
    distance_to_recovery_zone: float
    distance_to_goal: float


@dataclass
class RecoveryZoneDetector:
    """
    Detects whether the cube has left the normal operating region.

    The recovery zone is a known-good state distribution:
    for this first MVP, it is simply a circle around the cube's initial position.
    """

    recovery_center_xy: np.ndarray
    recovered_radius: float = 0.05
    unexpected_radius: float = 0.13
    goal_radius: float = 0.05

    @classmethod
    def from_initial_state(
        cls,
        state: PushCubeState,
        recovered_radius: float = 0.05,
        unexpected_radius: float = 0.13,
        goal_radius: float = 0.05,
    ) -> "RecoveryZoneDetector":
        return cls(
            recovery_center_xy=state.cube_xy.copy(),
            recovered_radius=recovered_radius,
            unexpected_radius=unexpected_radius,
            goal_radius=goal_radius,
        )

    def classify(self, state: PushCubeState) -> RecoveryDecision:
        dist_to_recovery = float(
            np.linalg.norm(state.cube_xy - self.recovery_center_xy)
        )
        dist_to_goal = float(np.linalg.norm(state.cube_xy - state.goal_xy))

        if dist_to_goal <= self.goal_radius:
            return RecoveryDecision(
                status=RecoveryStatus.TASK_LIKELY_DONE,
                reason="Cube is already close to the task goal.",
                distance_to_recovery_zone=dist_to_recovery,
                distance_to_goal=dist_to_goal,
            )

        if dist_to_recovery <= self.recovered_radius:
            return RecoveryDecision(
                status=RecoveryStatus.RECOVERED,
                reason="Cube is inside the known-good recovery zone.",
                distance_to_recovery_zone=dist_to_recovery,
                distance_to_goal=dist_to_goal,
            )

        if dist_to_recovery >= self.unexpected_radius:
            return RecoveryDecision(
                status=RecoveryStatus.NEEDS_RECOVERY,
                reason="Cube is outside the normal operating region.",
                distance_to_recovery_zone=dist_to_recovery,
                distance_to_goal=dist_to_goal,
            )

        return RecoveryDecision(
            status=RecoveryStatus.NORMAL,
            reason="Cube is between recovered and unexpected thresholds.",
            distance_to_recovery_zone=dist_to_recovery,
            distance_to_goal=dist_to_goal,
        )