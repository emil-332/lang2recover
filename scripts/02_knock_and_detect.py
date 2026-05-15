from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from lang2recover.recovery.detector import RecoveryStatus, RecoveryZoneDetector
from lang2recover.recovery.perturbations import knock_cube_by_xy_offset
from lang2recover.sim.maniskill_pushcube import read_pushcube_state


def _done(x) -> bool:
    try:
        return bool(x.item())
    except AttributeError:
        return bool(x)


def render_topdown_frame(
    cube_xy: np.ndarray,
    goal_xy: np.ndarray,
    recovery_center_xy: np.ndarray,
    status: str,
    step: int,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    # Plot zones.
    recovery_circle = plt.Circle(
        recovery_center_xy,
        0.05,
        fill=False,
        linewidth=2,
        label="Recovery zone",
    )
    unexpected_circle = plt.Circle(
        recovery_center_xy,
        0.13,
        fill=False,
        linestyle="--",
        linewidth=1,
        label="Unexpected threshold",
    )
    goal_circle = plt.Circle(
        goal_xy,
        0.05,
        fill=False,
        linewidth=2,
        label="Goal zone",
    )

    ax.add_patch(recovery_circle)
    ax.add_patch(unexpected_circle)
    ax.add_patch(goal_circle)

    # Plot points.
    ax.scatter([cube_xy[0]], [cube_xy[1]], s=120, label="Cube")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=120, marker="*", label="Goal")
    ax.scatter(
        [recovery_center_xy[0]],
        [recovery_center_xy[1]],
        s=80,
        marker="x",
        label="Recovery center",
    )

    # Keep a stable visible workspace.
    all_xy = np.vstack([cube_xy, goal_xy, recovery_center_xy])
    center = all_xy.mean(axis=0)
    span = 0.45
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.set_title(f"Lang2Recover MVP | step={step} | status={status}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    output_dir = Path("videos/02_knock_and_detect_topdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "knock_and_detect_topdown.mp4"

    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=1)

    initial_state = read_pushcube_state(env)
    detector = RecoveryZoneDetector.from_initial_state(
        initial_state,
        recovered_radius=0.05,
        unexpected_radius=0.13,
        goal_radius=0.05,
    )

    print("Initial cube xy:", initial_state.cube_xy)
    print("Goal xy:", initial_state.goal_xy)
    print("Recovery center xy:", detector.recovery_center_xy)

    frames = []
    knock_step = 35
    already_knocked = False

    for step in range(100):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        obs, reward, terminated, truncated, info = env.step(action)

        if step == knock_step and not already_knocked:
            print(f"\nStep {step}: applying artificial external perturbation.")
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True

        state = read_pushcube_state(env)
        decision = detector.classify(state)

        frames.append(
            render_topdown_frame(
                cube_xy=state.cube_xy,
                goal_xy=state.goal_xy,
                recovery_center_xy=detector.recovery_center_xy,
                status=decision.status.value,
                step=step,
            )
        )

        if step % 10 == 0 or decision.status == RecoveryStatus.NEEDS_RECOVERY:
            print(
                f"step={step:03d} | "
                f"status={decision.status.value:>15} | "
                f"dist_recovery={decision.distance_to_recovery_zone:.3f} | "
                f"dist_goal={decision.distance_to_goal:.3f} | "
                f"reason={decision.reason}"
            )

        if decision.status == RecoveryStatus.NEEDS_RECOVERY:
            print("\nRecovery trigger fired.")
            print("This is where the RL recovery policy will take over later.")
            break

        if _done(terminated) or _done(truncated):
            break

    env.close()

    imageio.mimsave(video_path, frames, fps=10)
    print(f"\nDone. Top-down video saved to: {video_path}")


if __name__ == "__main__":
    main()