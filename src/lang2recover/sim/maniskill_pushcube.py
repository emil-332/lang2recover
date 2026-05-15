from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PushCubeState:
    cube_pos: np.ndarray
    goal_pos: np.ndarray
    tcp_pos: np.ndarray

    @property
    def cube_xy(self) -> np.ndarray:
        return self.cube_pos[:2]

    @property
    def goal_xy(self) -> np.ndarray:
        return self.goal_pos[:2]

    @property
    def tcp_xy(self) -> np.ndarray:
        return self.tcp_pos[:2]


def _to_single_numpy(x: Any) -> np.ndarray:
    """Convert ManiSkill torch tensors to a single unbatched numpy vector."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)

    if arr.ndim >= 2 and arr.shape[0] == 1:
        arr = arr[0]

    return arr.astype(np.float32)


def get_base_env(env: Any) -> Any:
    """Return the underlying ManiSkill task env under Gymnasium/wrapper layers."""
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    return env


def read_pushcube_state(env: Any) -> PushCubeState:
    """
    Read useful state from ManiSkill PushCube-v1.

    This intentionally uses task internals for the MVP:
    - env.unwrapped.obj
    - env.unwrapped.goal_region
    - env.unwrapped.agent.tcp

    Later we can replace this with a cleaner observation-based interface.
    """
    base = get_base_env(env)

    required = ["obj", "goal_region", "agent"]
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise AttributeError(
            f"This does not look like a PushCube-style ManiSkill env. "
            f"Missing attributes: {missing}"
        )

    cube_pos = _to_single_numpy(base.obj.pose.p)
    goal_pos = _to_single_numpy(base.goal_region.pose.p)
    tcp_pos = _to_single_numpy(base.agent.tcp.pose.p)

    return PushCubeState(
        cube_pos=cube_pos,
        goal_pos=goal_pos,
        tcp_pos=tcp_pos,
    )