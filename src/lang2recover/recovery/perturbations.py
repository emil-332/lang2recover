from __future__ import annotations

from typing import Any, Sequence

import torch
from mani_skill.utils.structs.pose import Pose

from lang2recover.sim.maniskill_pushcube import get_base_env


def knock_cube_by_xy_offset(
    env: Any,
    offset_xy: Sequence[float] = (0.0, -0.18),
) -> None:
    """
    Artificially knock the cube away by teleporting it in XY.

    This is intentionally simple for the MVP:
    it simulates an external disturbance such as a bump, slip, or failed contact.

    Later we can replace teleportation with a physical impulse.
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

    if hasattr(obj, "set_linear_velocity"):
        obj.set_linear_velocity(torch.zeros_like(p))

    if hasattr(obj, "set_angular_velocity"):
        angular_velocity = torch.zeros((*p.shape[:-1], 3), dtype=p.dtype, device=p.device)
        obj.set_angular_velocity(angular_velocity)