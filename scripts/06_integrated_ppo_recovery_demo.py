from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from lang2recover.envs.recovery_2d_env import Recovery2DConfig
from lang2recover.recovery.detector import RecoveryStatus, RecoveryZoneDetector
from lang2recover.recovery.perturbations import knock_cube_by_xy_offset, set_cube_xy
from lang2recover.recovery.ppo_adapter import (
    build_recovery_2d_observation_from_maniskill_state,
    ppo_action_to_world_delta,
)
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
    action: np.ndarray | None,
    reward: float | None,
    distance_to_recovery: float,
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
        label="Task goal",
    )

    ax.add_patch(recovery_circle)
    ax.add_patch(unexpected_circle)
    ax.add_patch(goal_circle)

    ax.scatter([cube_xy[0]], [cube_xy[1]], s=160, label="Cube")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=150, marker="*", label="Task goal")
    ax.scatter(
        [recovery_center_xy[0]],
        [recovery_center_xy[1]],
        s=100,
        marker="x",
        label="Recovery center",
    )

    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        ax.arrow(
            cube_xy[0],
            cube_xy[1],
            action[0] * 0.045,
            action[1] * 0.045,
            head_width=0.01,
            length_includes_head=True,
            label="PPO action",
        )

    center = np.vstack([cube_xy, goal_xy, recovery_center_xy]).mean(axis=0)
    span = 0.45
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    reward_text = "n/a" if reward is None else f"{reward:.3f}"

    ax.set_title(
        f"Lang2Recover Integrated PPO Demo | step={step}\n"
        f"phase={phase} | status={status} | "
        f"dist={distance_to_recovery:.3f} | reward={reward_text}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    model_path = Path("checkpoints/recovery_ppo_2d.zip")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. "
            "Run scripts/04_train_recovery_ppo.py first."
        )

    output_dir = Path("videos/06_integrated_ppo_recovery_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "integrated_ppo_recovery_demo.mp4"

    recovery_config = Recovery2DConfig()
    model = PPO.load(model_path)

    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=3)

    initial_state = read_pushcube_state(env)
    detector = RecoveryZoneDetector.from_initial_state(
        initial_state,
        recovered_radius=recovery_config.recovered_radius,
        unexpected_radius=0.13,
        goal_radius=0.05,
    )

    print("Initial cube xy:", initial_state.cube_xy)
    print("Goal xy:", initial_state.goal_xy)
    print("Recovery center xy:", detector.recovery_center_xy)
    print("Loaded PPO policy from:", model_path)

    frames: list[np.ndarray] = []

    knock_step = 30
    already_knocked = False
    recovery_active = False
    resume_printed = False

    previous_state = read_pushcube_state(env)

    for step in range(120):
        # The normal policy is still a placeholder. For now it does nothing.
        # Later this becomes a VLA / imitation / scripted normal policy.
        normal_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        obs, env_reward, terminated, truncated, info = env.step(normal_action)

        if step == knock_step and not already_knocked:
            print(f"\nStep {step}: artificial disturbance knocks cube away.")
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True

        current_state = read_pushcube_state(env)
        decision = detector.classify(current_state)

        phase = "normal_policy"
        ppo_action = None

        if decision.status == RecoveryStatus.NEEDS_RECOVERY:
            recovery_active = True

        if recovery_active and decision.status != RecoveryStatus.RECOVERED:
            phase = "ppo_recovery"

            recovery_obs = build_recovery_2d_observation_from_maniskill_state(
                state=current_state,
                recovery_center_xy=detector.recovery_center_xy,
            )

            raw_action, _ = model.predict(recovery_obs, deterministic=True)
            recovery_action = ppo_action_to_world_delta(
                raw_action,
                config=recovery_config,
            )

            new_cube_xy = current_state.cube_xy + recovery_action.delta_xy
            set_cube_xy(env, new_cube_xy)

            ppo_action = recovery_action.action

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        reward_result = compute_recovery_reward(
            previous_state=previous_state,
            current_state=current_state,
            recovery_center_xy=detector.recovery_center_xy,
            action=ppo_action,
        )

        frames.append(
            render_topdown_frame(
                cube_xy=current_state.cube_xy,
                goal_xy=current_state.goal_xy,
                recovery_center_xy=detector.recovery_center_xy,
                status=decision.status.value,
                phase=phase,
                step=step,
                action=ppo_action,
                reward=reward_result.reward,
                distance_to_recovery=decision.distance_to_recovery_zone,
            )
        )

        if step % 10 == 0 or phase == "ppo_recovery" or decision.status == RecoveryStatus.RECOVERED:
            action_text = "None" if ppo_action is None else np.round(ppo_action, 3)
            print(
                f"step={step:03d} | "
                f"phase={phase:>14} | "
                f"status={decision.status.value:>15} | "
                f"dist_recovery={decision.distance_to_recovery_zone:.3f} | "
                f"ppo_action={action_text} | "
                f"reward={reward_result.reward:.3f}"
            )

        if recovery_active and decision.status == RecoveryStatus.RECOVERED:
            if not resume_printed:
                print("\nCube recovered.")
                print("Recovery policy hands control back to the normal VLA-style policy.")
                resume_printed = True

            # Add a few extra frames after recovery so the final state is visible.
            if step > knock_step + 8:
                break

        previous_state = current_state

        if _done(terminated) or _done(truncated):
            break

    env.close()

    imageio.mimsave(video_path, frames, fps=10)

    print("\nIntegrated PPO recovery demo finished.")
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()