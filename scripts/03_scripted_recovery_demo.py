from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from lang2recover.recovery.detector import RecoveryStatus, RecoveryZoneDetector
from lang2recover.recovery.perturbations import knock_cube_by_xy_offset, set_cube_xy
from lang2recover.rewards.generated_recovery_reward import compute_recovery_reward
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
    phase: str,
    step: int,
    reward: float | None,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

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

    ax.scatter([cube_xy[0]], [cube_xy[1]], s=140, label="Cube")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=140, marker="*", label="Task goal")
    ax.scatter(
        [recovery_center_xy[0]],
        [recovery_center_xy[1]],
        s=90,
        marker="x",
        label="Recovery center",
    )

    center = np.vstack([cube_xy, goal_xy, recovery_center_xy]).mean(axis=0)
    span = 0.45
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    reward_text = "n/a" if reward is None else f"{reward:.3f}"
    ax.set_title(
        f"Lang2Recover | step={step} | phase={phase}\n"
        f"status={status} | reward={reward_text}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    output_dir = Path("videos/03_scripted_recovery_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "scripted_recovery_demo.mp4"

    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=2)

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

    knock_step = 30
    recovery_steps = 35
    already_knocked = False
    recovery_started = False
    recovery_start_xy: np.ndarray | None = None

    previous_state = read_pushcube_state(env)

    for step in range(100):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        obs, reward, terminated, truncated, info = env.step(action)

        if step == knock_step and not already_knocked:
            print(f"\nStep {step}: applying artificial external perturbation.")
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True

        current_state = read_pushcube_state(env)
        decision = detector.classify(current_state)

        phase = "normal"

        if decision.status == RecoveryStatus.NEEDS_RECOVERY and not recovery_started:
            recovery_started = True
            recovery_start_xy = current_state.cube_xy.copy()
            print("\nRecovery trigger fired.")
            print("Starting scripted recovery placeholder.")

        if recovery_started:
            phase = "scripted_recovery"

            assert recovery_start_xy is not None

            recovery_progress = min(1.0, (step - knock_step) / recovery_steps)

            new_cube_xy = (
                (1.0 - recovery_progress) * recovery_start_xy
                + recovery_progress * detector.recovery_center_xy
            )

            set_cube_xy(env, new_cube_xy)
            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        reward_result = compute_recovery_reward(
            previous_state=previous_state,
            current_state=current_state,
            recovery_center_xy=detector.recovery_center_xy,
            action=action,
        )

        frames.append(
            render_topdown_frame(
                cube_xy=current_state.cube_xy,
                goal_xy=current_state.goal_xy,
                recovery_center_xy=detector.recovery_center_xy,
                status=decision.status.value,
                phase=phase,
                step=step,
                reward=reward_result.reward,
            )
        )

        if step % 10 == 0 or decision.status in {
            RecoveryStatus.NEEDS_RECOVERY,
            RecoveryStatus.RECOVERED,
        }:
            print(
                f"step={step:03d} | "
                f"phase={phase:>18} | "
                f"status={decision.status.value:>15} | "
                f"dist_recovery={decision.distance_to_recovery_zone:.3f} | "
                f"reward={reward_result.reward:.3f} | "
                f"reason={decision.reason}"
            )

        previous_state = current_state

        if recovery_started and decision.status == RecoveryStatus.RECOVERED and step > knock_step + 5:
            print("\nCube recovered.")
            print("This is where the normal VLA policy would resume.")
            break

        if _done(terminated) or _done(truncated):
            break

    env.close()

    imageio.mimsave(video_path, frames, fps=10)
    print(f"\nDone. Scripted recovery video saved to: {video_path}")


if __name__ == "__main__":
    main()