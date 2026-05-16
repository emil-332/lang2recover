from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

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

    reward_mode:
        manual_dense:
            uses the built-in dense recovery reward.

        language_generated:
            loads generated_rewards/cube_recovery_language_reward.py and uses
            its compute_language_reward function.
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
    reward_mode: str = "manual_dense"
    generated_reward_path: str = "generated_rewards/cube_recovery_language_reward.py"


def _load_python_module_from_path(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(
            f"Generated reward file not found: {path}. "
            "Run scripts/07_generate_language_reward.py first."
        )

    spec = spec_from_file_location("cube_recovery_language_reward", path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        self.generated_reward_module: ModuleType | None = None

        if self.config.reward_mode == "language_generated":
            self.generated_reward_module = _load_python_module_from_path(
                Path(self.config.generated_reward_path)
            )

            if not hasattr(self.generated_reward_module, "compute_language_reward"):
                raise AttributeError(
                    "Generated reward module must define compute_language_reward."
                )

        valid_reward_modes = {"manual_dense", "language_generated"}
        if self.config.reward_mode not in valid_reward_modes:
            raise ValueError(
                f"Invalid reward_mode={self.config.reward_mode!r}. "
                f"Expected one of {sorted(valid_reward_modes)}."
            )

        high = np.array(
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
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

    def _compute_reward(
        self,
        previous_distance: float,
        current_distance: float,
        action: np.ndarray,
        is_recovered: bool,
        is_out_of_bounds: bool,
    ) -> float:
        action_norm = float(np.linalg.norm(action))

        if self.config.reward_mode == "manual_dense":
            progress = previous_distance - current_distance

            reward = 0.0
            reward += -self.config.distance_weight * current_distance
            reward += self.config.progress_weight * progress
            reward -= self.config.action_penalty_weight * action_norm

            if is_recovered:
                reward += self.config.success_bonus

            if is_out_of_bounds:
                reward -= self.config.out_of_bounds_penalty

            return float(reward)

        if self.config.reward_mode == "language_generated":
            assert self.generated_reward_module is not None

            return float(
                self.generated_reward_module.compute_language_reward(
                    previous_distance_to_recovery=previous_distance,
                    distance_to_recovery=current_distance,
                    action_norm=action_norm,
                    is_recovered=is_recovered,
                    is_out_of_bounds=is_out_of_bounds,
                )
            )

        raise RuntimeError(f"Unsupported reward mode: {self.config.reward_mode}")

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
            "reward_mode": self.config.reward_mode,
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
        is_recovered = self._is_recovered()
        is_out_of_bounds = self._is_out_of_bounds()

        reward = self._compute_reward(
            previous_distance=previous_distance,
            current_distance=current_distance,
            action=action,
            is_recovered=is_recovered,
            is_out_of_bounds=is_out_of_bounds,
        )

        terminated = False

        if is_recovered:
            terminated = True

        if is_out_of_bounds:
            terminated = True

        truncated = self.step_count >= self.config.max_episode_steps

        self.previous_distance = current_distance

        info = {
            "distance_to_recovery": current_distance,
            "progress": previous_distance - current_distance,
            "is_recovered": is_recovered,
            "is_out_of_bounds": is_out_of_bounds,
            "recovery_center_xy": self.recovery_center_xy.copy(),
            "goal_xy": self.goal_xy.copy(),
            "cube_xy": self.cube_xy.copy(),
            "step_count": self.step_count,
            "reward_mode": self.config.reward_mode,
        }

        return self._get_obs(), float(reward), terminated, truncated, info