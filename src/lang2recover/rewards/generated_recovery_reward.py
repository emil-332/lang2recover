from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lang2recover.sim.maniskill_pushcube import PushCubeState


@dataclass(frozen=True)
class RecoveryRewardConfig:
    """
    Reward configuration for the recovery task.

    Natural-language objective:
    "Move the displaced cube back into the known-good recovery zone while
    keeping the object stable and avoiding unnecessary motion."
    """

    distance_weight: float = 4.0
    progress_weight: float = 8.0
    success_bonus: float = 10.0
    action_penalty_weight: float = 0.01
    recovered_radius: float = 0.05


@dataclass(frozen=True)
class RecoveryRewardResult:
    reward: float
    distance_to_recovery_zone: float
    progress: float
    is_recovered: bool


def compute_recovery_reward(
    previous_state: PushCubeState,
    current_state: PushCubeState,
    recovery_center_xy: np.ndarray,
    action: np.ndarray | None = None,
    config: RecoveryRewardConfig = RecoveryRewardConfig(),
) -> RecoveryRewardResult:
    """
    Dense reward for recovering the cube to a known-good state.

    This reward encourages:
    1. small distance between cube and recovery zone,
    2. progress toward the recovery zone,
    3. a bonus once the cube is recovered,
    4. small actions.
    """
    prev_dist = float(
        np.linalg.norm(previous_state.cube_xy - recovery_center_xy)
    )
    curr_dist = float(
        np.linalg.norm(current_state.cube_xy - recovery_center_xy)
    )

    progress = prev_dist - curr_dist

    reward = 0.0
    reward += -config.distance_weight * curr_dist
    reward += config.progress_weight * progress

    is_recovered = curr_dist <= config.recovered_radius

    if is_recovered:
        reward += config.success_bonus

    if action is not None:
        reward -= config.action_penalty_weight * float(np.linalg.norm(action))

    return RecoveryRewardResult(
        reward=float(reward),
        distance_to_recovery_zone=curr_dist,
        progress=float(progress),
        is_recovered=is_recovered,
    )