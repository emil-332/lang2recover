from __future__ import annotations

from typing import Any, Sequence

import torch
from mani_skill.utils.structs.pose import Pose

from lang2recover.sim.maniskill_pushcube import get_base_env


def _zero_object_velocity(obj: Any, p: torch.Tensor) -> None:
    if hasattr(obj, "set_linear_velocity"):
        obj.set_linear_velocity(torch.zeros_like(p))

    if hasattr(obj, "set_angular_velocity"):
        angular_velocity = torch.zeros((*p.shape[:-1], 3), dtype=p.dtype, device=p.device)
        obj.set_angular_velocity(angular_velocity)


def set_cube_xy(
    env: Any,
    xy: Sequence[float],
) -> None:
    """
    Set the cube XY position directly.

    This is used only for scripted MVP demos and debugging.
    The final RL policy will move the cube through actions, not teleportation.
    """
    base = get_base_env(env)

    if not hasattr(base, "obj"):
        raise AttributeError("Expected env.unwrapped.obj to exist for PushCube-v1.")

    obj = base.obj
    current_pose = obj.pose

    p = current_pose.p.clone()
    q = current_pose.q.clone()

    target_xy = torch.tensor(
        xy,
        dtype=p.dtype,
        device=p.device,
    )

    p[..., :2] = target_xy

    obj.set_pose(Pose.create_from_pq(p=p, q=q))
    _zero_object_velocity(obj, p)


def knock_cube_by_xy_offset(
    env: Any,
    offset_xy: Sequence[float] = (0.0, -0.18),
) -> None:
    """
    Artificially knock the cube away by teleporting it in XY.

    This simulates an external disturbance such as a bump, slip, or failed contact.
    """
    base = get_base_env(env)

    if not hasattr(base, "obj"):
        raise AttributeError("Expected env.unwrapped.obj to exist for PushCube-v1.")

    obj = base.obj
    current_pose = obj.pose

    p = current_pose.p.clone()
    q = current_pose.q.clone()

    offset = torch.tensor(
        [offset_xy[0], offset_xy[1]],
        dtype=p.dtype,
        device=p.device,
    )

    p[..., :2] = p[..., :2] + offset

    obj.set_pose(Pose.create_from_pq(p=p, q=q))
    _zero_object_velocity(obj, p)