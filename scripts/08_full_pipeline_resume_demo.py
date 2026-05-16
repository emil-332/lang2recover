from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from lang2recover.envs.recovery_2d_env import Recovery2DConfig
from lang2recover.policies.cube_level_normal_policy import CubeLevelNormalPolicy
from lang2recover.recovery.detector import RecoveryStatus, RecoveryZoneDetector
from lang2recover.recovery.perturbations import knock_cube_by_xy_offset, set_cube_xy
from lang2recover.recovery.ppo_adapter import (
    build_recovery_2d_observation_from_maniskill_state,
    ppo_action_to_world_delta,
)
from lang2recover.rewards.generated_recovery_reward import compute_recovery_reward
from lang2recover.sim.maniskill_pushcube import read_pushcube_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="language_generated",
        choices=["manual_dense", "language_generated"],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--knock-step", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    return parser.parse_args()


def render_topdown_frame(
    cube_xy: np.ndarray,
    goal_xy: np.ndarray,
    recovery_center_xy: np.ndarray,
    status: str,
    phase: str,
    step: int,
    normal_label: str | None,
    ppo_action: np.ndarray | None,
    recovery_reward: float | None,
    distance_to_recovery: float,
    distance_to_goal: float,
    instruction: str,
    detector_enabled: bool,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)

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
        label="Recovery trigger threshold",
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

    ax.scatter([cube_xy[0]], [cube_xy[1]], s=170, label="Cube")
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=160, marker="*", label="Task goal")
    ax.scatter(
        [recovery_center_xy[0]],
        [recovery_center_xy[1]],
        s=110,
        marker="x",
        label="Recovery center",
    )

    if ppo_action is not None:
        ppo_action = np.asarray(ppo_action, dtype=np.float32)
        ax.arrow(
            cube_xy[0],
            cube_xy[1],
            ppo_action[0] * 0.045,
            ppo_action[1] * 0.045,
            head_width=0.01,
            length_includes_head=True,
        )

    all_xy = np.vstack([cube_xy, goal_xy, recovery_center_xy])
    center = all_xy.mean(axis=0)
    span = 0.45
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    reward_text = "n/a" if recovery_reward is None else f"{recovery_reward:.3f}"
    normal_text = "n/a" if normal_label is None else normal_label
    detector_text = "enabled" if detector_enabled else "disabled"

    ax.set_title(
        "Lang2Recover: Full Recovery → Resume Demo\n"
        f"step={step} | phase={phase} | detector={detector_text}\n"
        f"d_recovery={distance_to_recovery:.3f} | d_goal={distance_to_goal:.3f} | "
        f"recovery_reward={reward_text}"
    )

    ax.text(
        0.02,
        0.02,
        f'Instruction: "{instruction}"\n'
        f"Normal policy: {normal_text}\n"
        f"Recovery status: {status}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def resolve_model_path(reward_mode: str) -> Path:
    preferred = Path(f"checkpoints/recovery_ppo_2d_{reward_mode}.zip")

    if preferred.exists():
        return preferred

    if reward_mode == "manual_dense":
        legacy = Path("checkpoints/recovery_ppo_2d.zip")
        if legacy.exists():
            return legacy

    raise FileNotFoundError(
        f"Could not find PPO checkpoint for reward_mode={reward_mode!r}. "
        f"Expected: {preferred}. "
        f"Run: python scripts/04_train_recovery_ppo.py "
        f"--reward-mode {reward_mode} --timesteps 50000"
    )


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.reward_mode)
    model = PPO.load(model_path)

    output_dir = Path(f"videos/08_full_pipeline_resume_demo/{args.reward_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "full_pipeline_recovery_resume_demo.mp4"

    recovery_config = Recovery2DConfig(reward_mode=args.reward_mode)

    normal_policy = CubeLevelNormalPolicy(
        instruction="Move the cube to the task goal.",
        max_step_size=0.018,
        goal_radius=0.05,
    )

    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    env.reset(seed=args.seed)

    initial_state = read_pushcube_state(env)
    detector = RecoveryZoneDetector.from_initial_state(
        initial_state,
        recovered_radius=recovery_config.recovered_radius,
        unexpected_radius=0.13,
        goal_radius=normal_policy.goal_radius,
    )

    print("Instruction:", normal_policy.instruction)
    print("Initial cube xy:", initial_state.cube_xy)
    print("Goal xy:", initial_state.goal_xy)
    print("Recovery center xy:", detector.recovery_center_xy)
    print("Loaded PPO policy from:", model_path)
    print("Reward mode:", args.reward_mode)
    print("Knock step:", args.knock_step)

    frames: list[np.ndarray] = []

    already_knocked = False
    recovery_active = False
    recovery_finished_once = False
    task_success = False

    previous_state = read_pushcube_state(env)

    for step in range(args.max_steps):
        current_state = read_pushcube_state(env)
        decision = detector.classify(current_state)

        phase = "normal_policy"
        normal_label: str | None = None
        ppo_action: np.ndarray | None = None

        detector_enabled = already_knocked and not recovery_finished_once

        if step == args.knock_step and not already_knocked:
            print(f"\nStep {step}: artificial disturbance knocks cube away.")
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True
            detector_enabled = True

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        if detector_enabled and decision.status == RecoveryStatus.NEEDS_RECOVERY:
            recovery_active = True

        if recovery_active:
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

            if decision.status == RecoveryStatus.RECOVERED:
                recovery_active = False
                recovery_finished_once = True
                detector_enabled = False
                phase = "resume_normal_policy"
                print("\nCube recovered.")
                print("Control returns to the normal VLA-style policy.")

        elif not task_success:
            phase = "normal_policy"

            normal_action = normal_policy.act(current_state)
            normal_label = normal_action.action_label

            if not normal_policy.is_task_complete(current_state):
                new_cube_xy = current_state.cube_xy + normal_action.delta_xy
                set_cube_xy(env, new_cube_xy)

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        recovery_reward_result = compute_recovery_reward(
            previous_state=previous_state,
            current_state=current_state,
            recovery_center_xy=detector.recovery_center_xy,
            action=ppo_action,
        )

        distance_to_goal = float(
            np.linalg.norm(current_state.cube_xy - current_state.goal_xy)
        )
        distance_to_recovery = decision.distance_to_recovery_zone

        if normal_policy.is_task_complete(current_state):
            task_success = True
            phase = "task_success"

        frames.append(
            render_topdown_frame(
                cube_xy=current_state.cube_xy,
                goal_xy=current_state.goal_xy,
                recovery_center_xy=detector.recovery_center_xy,
                status=decision.status.value,
                phase=phase,
                step=step,
                normal_label=normal_label,
                ppo_action=ppo_action,
                recovery_reward=recovery_reward_result.reward,
                distance_to_recovery=distance_to_recovery,
                distance_to_goal=distance_to_goal,
                instruction=normal_policy.instruction,
                detector_enabled=detector_enabled,
            )
        )

        if (
            step % 2 == 0
            or phase in {"ppo_recovery", "resume_normal_policy", "task_success"}
        ):
            action_text = "None" if ppo_action is None else np.round(ppo_action, 3)
            print(
                f"step={step:03d} | "
                f"phase={phase:>20} | "
                f"detector={'on' if detector_enabled else 'off':>3} | "
                f"status={decision.status.value:>15} | "
                f"d_recovery={distance_to_recovery:.3f} | "
                f"d_goal={distance_to_goal:.3f} | "
                f"ppo_action={action_text} | "
                f"normal={normal_label}"
            )

        previous_state = current_state

        if task_success and recovery_finished_once:
            print("\nTask completed after PPO recovery and normal-policy resume.")
            for _ in range(8):
                frames.append(frames[-1])
            break

    env.close()

    imageio.mimsave(video_path, frames, fps=10)

    print("\nFull pipeline demo finished.")
    print(f"Recovered at least once: {recovery_finished_once}")
    print(f"Task success: {task_success}")
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()