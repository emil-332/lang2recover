from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lang2recover.sim.maniskill_pushcube import PushCubeState


@dataclass(frozen=True)
class CubeLevelAction:
    """
    Cube-level action used by the lightweight normal-policy placeholder.

    This is not yet robot joint control. It represents the intended cube motion
    of a normal VLA-style task policy.
    """

    delta_xy: np.ndarray
    action_label: str


@dataclass
class CubeLevelNormalPolicy:
    """
    Lightweight normal task policy.

    This is a VLA-style placeholder:
    given the instruction "move cube to goal", it moves the cube directly toward
    the task goal at the cube-state level.

    Later this can be replaced by:
    - scripted robot policy,
    - imitation policy,
    - SmolVLA wrapper,
    - OpenVLA wrapper.
    """

    instruction: str = "Move the cube to the task goal."
    max_step_size: float = 0.018
    goal_radius: float = 0.05

    def is_task_complete(self, state: PushCubeState) -> bool:
        distance_to_goal = float(np.linalg.norm(state.cube_xy - state.goal_xy))
        return distance_to_goal <= self.goal_radius

    def act(self, state: PushCubeState) -> CubeLevelAction:
        direction = state.goal_xy - state.cube_xy
        distance = float(np.linalg.norm(direction))

        if distance <= self.goal_radius:
            return CubeLevelAction(
                delta_xy=np.zeros(2, dtype=np.float32),
                action_label="task_complete",
            )

        if distance < 1e-8:
            unit_direction = np.zeros(2, dtype=np.float32)
        else:
            unit_direction = (direction / distance).astype(np.float32)

        step_size = min(self.max_step_size, distance)
        delta_xy = unit_direction * np.float32(step_size)

        return CubeLevelAction(
            delta_xy=delta_xy.astype(np.float32),
            action_label="move_toward_goal",
        )