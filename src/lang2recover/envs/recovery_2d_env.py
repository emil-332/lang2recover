from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Recovery2DConfig:
    """
    Lightweight 2D recovery environment.

    Natural-language objective:
    "Move the displaced cube back into the known-good recovery zone while
    avoiding unnecessary motion."

    This is the first RL milestone. It trains the recovery logic cheaply before
    we connect it back to full ManiSkill robot control.
    """

    max_episode_steps: int = 60
    recovered_radius: float = 0.05
    workspace_radius: float = 0.35
    min_start_distance: float = 0.14
    max_start_distance: float = 0.26
    max_action_step: float = 0.025
    action_penalty_weight: float = 0.02
    distance_weight: float = 3.0
    progress_weight: float = 8.0
    success_bonus: float = 10.0
    out_of_bounds_penalty: float = 5.0
    motion_noise_std: float = 0.001


class Recovery2DEnv(gym.Env):
    """
    A small Gymnasium environment for learning recovery behavior.

    State:
        cube_xy
        recovery_center_xy
        goal_xy
        cube_minus_recovery
        cube_minus_goal

    Action:
        2D delta direction in [-1, 1]^2

    Goal:
        move cube_xy into the recovery zone.

    This environment does not simulate a robot arm yet. It trains the core
    recovery policy cheaply and gives us a real PPO baseline.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Recovery2DConfig | None = None):
        super().__init__()

        self.config = config or Recovery2DConfig()

        high = np.array(
            [
                1.0,
                1.0,  # cube xy
                1.0,
                1.0,  # recovery center xy
                1.0,
                1.0,  # task goal xy
                1.0,
                1.0,  # cube - recovery
                1.0,
                1.0,  # cube - goal
            ],
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            shape=(10,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.recovery_center_xy = np.zeros(2, dtype=np.float32)
        self.goal_xy = np.array([0.20, 0.0], dtype=np.float32)
        self.cube_xy = np.zeros(2, dtype=np.float32)

        self.previous_distance = 0.0
        self.step_count = 0

    def _sample_displaced_cube_xy(self) -> np.ndarray:
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        radius = self.np_random.uniform(
            self.config.min_start_distance,
            self.config.max_start_distance,
        )

        offset = np.array(
            [np.cos(angle), np.sin(angle)],
            dtype=np.float32,
        ) * np.float32(radius)

        return self.recovery_center_xy + offset

    def _get_obs(self) -> np.ndarray:
        cube_minus_recovery = self.cube_xy - self.recovery_center_xy
        cube_minus_goal = self.cube_xy - self.goal_xy

        obs = np.concatenate(
            [
                self.cube_xy,
                self.recovery_center_xy,
                self.goal_xy,
                cube_minus_recovery,
                cube_minus_goal,
            ],
            dtype=np.float32,
        )

        return obs.astype(np.float32)

    def _distance_to_recovery(self) -> float:
        return float(np.linalg.norm(self.cube_xy - self.recovery_center_xy))

    def _is_recovered(self) -> bool:
        return self._distance_to_recovery() <= self.config.recovered_radius

    def _is_out_of_bounds(self) -> bool:
        return (
            float(np.linalg.norm(self.cube_xy - self.recovery_center_xy))
            > self.config.workspace_radius
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        self.step_count = 0

        self.recovery_center_xy = np.zeros(2, dtype=np.float32)
        self.goal_xy = np.array([0.20, 0.0], dtype=np.float32)
        self.cube_xy = self._sample_displaced_cube_xy().astype(np.float32)

        self.previous_distance = self._distance_to_recovery()

        info = {
            "distance_to_recovery": self.previous_distance,
            "is_recovered": self._is_recovered(),
            "recovery_center_xy": self.recovery_center_xy.copy(),
            "goal_xy": self.goal_xy.copy(),
            "cube_xy": self.cube_xy.copy(),
        }

        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        previous_distance = self._distance_to_recovery()

        delta = action * self.config.max_action_step

        if self.config.motion_noise_std > 0.0:
            noise = self.np_random.normal(
                loc=0.0,
                scale=self.config.motion_noise_std,
                size=(2,),
            ).astype(np.float32)
        else:
            noise = np.zeros(2, dtype=np.float32)

        self.cube_xy = (self.cube_xy + delta + noise).astype(np.float32)

        current_distance = self._distance_to_recovery()
        progress = previous_distance - current_distance

        reward = 0.0
        reward += -self.config.distance_weight * current_distance
        reward += self.config.progress_weight * progress
        reward -= self.config.action_penalty_weight * float(np.linalg.norm(action))

        terminated = False

        if self._is_recovered():
            reward += self.config.success_bonus
            terminated = True

        if self._is_out_of_bounds():
            reward -= self.config.out_of_bounds_penalty
            terminated = True

        truncated = self.step_count >= self.config.max_episode_steps

        self.previous_distance = current_distance

        info = {
            "distance_to_recovery": current_distance,
            "progress": progress,
            "is_recovered": self._is_recovered(),
            "is_out_of_bounds": self._is_out_of_bounds(),
            "recovery_center_xy": self.recovery_center_xy.copy(),
            "goal_xy": self.goal_xy.copy(),
            "cube_xy": self.cube_xy.copy(),
            "step_count": self.step_count,
        }

        return self._get_obs(), float(reward), terminated, truncated, info