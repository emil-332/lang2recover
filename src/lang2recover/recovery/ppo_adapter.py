from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lang2recover.envs.recovery_2d_env import Recovery2DConfig
from lang2recover.sim.maniskill_pushcube import PushCubeState


@dataclass(frozen=True)
class PPORecoveryAction:
    """
    Action predicted by the 2D PPO recovery policy.

    action:
        raw PPO action in [-1, 1]^2

    delta_xy:
        actual XY displacement applied to the cube in world coordinates
    """

    action: np.ndarray
    delta_xy: np.ndarray


def build_recovery_2d_observation_from_maniskill_state(
    state: PushCubeState,
    recovery_center_xy: np.ndarray,
) -> np.ndarray:
    """
    Convert ManiSkill world coordinates into the observation format expected by
    Recovery2DEnv.

    The PPO policy was trained in a local coordinate frame where:
    - recovery_center_xy = [0, 0]
    - cube_xy is relative to the recovery center
    - goal_xy is relative to the recovery center

    ManiSkill uses world coordinates, so we transform into the same local frame.
    """
    recovery_center_xy = np.asarray(recovery_center_xy, dtype=np.float32)

    cube_xy_local = (state.cube_xy - recovery_center_xy).astype(np.float32)
    recovery_center_local = np.zeros(2, dtype=np.float32)
    goal_xy_local = (state.goal_xy - recovery_center_xy).astype(np.float32)

    cube_minus_recovery = cube_xy_local - recovery_center_local
    cube_minus_goal = cube_xy_local - goal_xy_local

    obs = np.concatenate(
        [
            cube_xy_local,
            recovery_center_local,
            goal_xy_local,
            cube_minus_recovery,
            cube_minus_goal,
        ],
        dtype=np.float32,
    )

    return obs.astype(np.float32)


def ppo_action_to_world_delta(
    action: np.ndarray,
    config: Recovery2DConfig | None = None,
) -> PPORecoveryAction:
    """
    Convert a PPO action from [-1, 1]^2 into a world-space XY delta.

    This mirrors Recovery2DEnv.step(...), where:
        delta = action * max_action_step
    """
    config = config or Recovery2DConfig()

    action = np.asarray(action, dtype=np.float32)
    action = np.clip(action, -1.0, 1.0)

    delta_xy = action * np.float32(config.max_action_step)

    return PPORecoveryAction(
        action=action,
        delta_xy=delta_xy.astype(np.float32),
    )