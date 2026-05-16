from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from lang2recover.sim.maniskill_pushcube import read_pushcube_state


Strategy = Literal["no_recovery", "ppo_recovery"]


@dataclass(frozen=True)
class FrameState:
    step: int
    strategy: str
    phase: str
    cube_xy: np.ndarray
    goal_xy: np.ndarray
    recovery_center_xy: np.ndarray
    distance_to_goal: float
    distance_to_recovery: float
    status: str
    detector_enabled: bool
    ppo_action: np.ndarray | None
    task_success: bool
    recovered_once: bool


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
    parser.add_argument("--max-steps", type=int, default=70)
    return parser.parse_args()


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


def run_strategy_rollout(
    strategy: Strategy,
    model: PPO | None,
    reward_mode: str,
    seed: int,
    knock_step: int,
    max_steps: int,
) -> list[FrameState]:
    recovery_config = Recovery2DConfig(reward_mode=reward_mode)

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

    env.reset(seed=seed)

    initial_state = read_pushcube_state(env)
    detector = RecoveryZoneDetector.from_initial_state(
        initial_state,
        recovered_radius=recovery_config.recovered_radius,
        unexpected_radius=0.13,
        goal_radius=normal_policy.goal_radius,
    )

    frames: list[FrameState] = []

    already_knocked = False
    recovery_active = False
    recovered_once = False
    task_success = False

    for step in range(max_steps):
        current_state = read_pushcube_state(env)
        decision = detector.classify(current_state)

        phase = "normal_policy"
        ppo_action: np.ndarray | None = None

        detector_enabled = already_knocked and not recovered_once

        if step == knock_step and not already_knocked:
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True
            detector_enabled = True

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        if detector_enabled and decision.status == RecoveryStatus.NEEDS_RECOVERY:
            recovery_active = True

        if recovery_active and not recovered_once:
            if strategy == "no_recovery":
                phase = "failed_no_recovery"
                # Deliberately do nothing:
                # the normal policy is assumed unreliable outside its expected
                # state distribution, so the baseline stalls.

            elif strategy == "ppo_recovery":
                phase = "ppo_recovery"

                if model is None:
                    raise ValueError("PPO model is required for ppo_recovery.")

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
                    recovered_once = True
                    detector_enabled = False
                    phase = "resume_normal_policy"

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        elif not task_success:
            phase = "normal_policy"

            normal_action = normal_policy.act(current_state)

            if not normal_policy.is_task_complete(current_state):
                new_cube_xy = current_state.cube_xy + normal_action.delta_xy
                set_cube_xy(env, new_cube_xy)

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

        distance_to_goal = float(np.linalg.norm(current_state.cube_xy - current_state.goal_xy))
        distance_to_recovery = float(
            np.linalg.norm(current_state.cube_xy - detector.recovery_center_xy)
        )

        if normal_policy.is_task_complete(current_state):
            task_success = True
            phase = "task_success"

        frames.append(
            FrameState(
                step=step,
                strategy=strategy,
                phase=phase,
                cube_xy=current_state.cube_xy.copy(),
                goal_xy=current_state.goal_xy.copy(),
                recovery_center_xy=detector.recovery_center_xy.copy(),
                distance_to_goal=distance_to_goal,
                distance_to_recovery=distance_to_recovery,
                status=decision.status.value,
                detector_enabled=detector_enabled,
                ppo_action=None if ppo_action is None else ppo_action.copy(),
                task_success=task_success,
                recovered_once=recovered_once,
            )
        )

        if task_success and recovered_once:
            for _ in range(8):
                frames.append(frames[-1])
            break

    env.close()

    return frames


def pad_frames(frames: list[FrameState], target_len: int) -> list[FrameState]:
    if len(frames) >= target_len:
        return frames

    return frames + [frames[-1]] * (target_len - len(frames))


def draw_panel(ax, frame: FrameState, title: str) -> None:
    recovery_circle = plt.Circle(
        frame.recovery_center_xy,
        0.05,
        fill=False,
        linewidth=2,
        label="Recovery zone",
    )
    trigger_circle = plt.Circle(
        frame.recovery_center_xy,
        0.13,
        fill=False,
        linestyle="--",
        linewidth=1,
        label="Recovery trigger",
    )
    goal_circle = plt.Circle(
        frame.goal_xy,
        0.05,
        fill=False,
        linewidth=2,
        label="Task goal",
    )

    ax.add_patch(recovery_circle)
    ax.add_patch(trigger_circle)
    ax.add_patch(goal_circle)

    ax.scatter([frame.cube_xy[0]], [frame.cube_xy[1]], s=170, label="Cube")
    ax.scatter([frame.goal_xy[0]], [frame.goal_xy[1]], s=150, marker="*", label="Goal")
    ax.scatter(
        [frame.recovery_center_xy[0]],
        [frame.recovery_center_xy[1]],
        s=100,
        marker="x",
        label="Recovery center",
    )

    if frame.ppo_action is not None:
        ax.arrow(
            frame.cube_xy[0],
            frame.cube_xy[1],
            frame.ppo_action[0] * 0.045,
            frame.ppo_action[1] * 0.045,
            head_width=0.01,
            length_includes_head=True,
        )

    all_xy = np.vstack([frame.cube_xy, frame.goal_xy, frame.recovery_center_xy])
    center = all_xy.mean(axis=0)
    span = 0.45

    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    detector_text = "on" if frame.detector_enabled else "off"
    success_text = "yes" if frame.task_success else "no"
    recovered_text = "yes" if frame.recovered_once else "no"

    ax.set_title(
        f"{title}\n"
        f"phase={frame.phase} | detector={detector_text}\n"
        f"d_goal={frame.distance_to_goal:.3f} | "
        f"recovered={recovered_text} | success={success_text}",
        fontsize=10,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")


def render_comparison_frame(
    left: FrameState,
    right: FrameState,
    global_step: int,
) -> np.ndarray:
    # 1280x640, divisible by 16, good for video codecs.
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4), dpi=100)

    draw_panel(axes[0], left, "Baseline: No Recovery")
    draw_panel(axes[1], right, "Lang2Recover: PPO Recovery")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
    )

    fig.suptitle(
        "Lang2Recover Comparison: Unexpected State Recovery",
        fontsize=14,
        y=0.98,
    )

    fig.text(
        0.5,
        0.035,
        f"global_step={global_step} | "
        "normal policy → disturbance → recovery if available → resume",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.93))

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)

    return frame


def main() -> None:
    args = parse_args()

    model_path = resolve_model_path(args.reward_mode)
    model = PPO.load(model_path)

    print("Creating strategy comparison video.")
    print("Reward mode:", args.reward_mode)
    print("Loaded PPO checkpoint:", model_path)
    print("Seed:", args.seed)
    print("Knock step:", args.knock_step)

    no_recovery_frames = run_strategy_rollout(
        strategy="no_recovery",
        model=None,
        reward_mode=args.reward_mode,
        seed=args.seed,
        knock_step=args.knock_step,
        max_steps=args.max_steps,
    )

    ppo_recovery_frames = run_strategy_rollout(
        strategy="ppo_recovery",
        model=model,
        reward_mode=args.reward_mode,
        seed=args.seed,
        knock_step=args.knock_step,
        max_steps=args.max_steps,
    )

    max_len = max(len(no_recovery_frames), len(ppo_recovery_frames))
    no_recovery_frames = pad_frames(no_recovery_frames, max_len)
    ppo_recovery_frames = pad_frames(ppo_recovery_frames, max_len)

    output_dir = Path(f"videos/10_strategy_comparison/{args.reward_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "no_recovery_vs_ppo_recovery.mp4"

    rendered_frames = []

    for idx, (left, right) in enumerate(zip(no_recovery_frames, ppo_recovery_frames)):
        rendered_frames.append(
            render_comparison_frame(
                left=left,
                right=right,
                global_step=idx,
            )
        )

    imageio.mimsave(video_path, rendered_frames, fps=10)

    no_recovery_final = no_recovery_frames[-1]
    ppo_final = ppo_recovery_frames[-1]

    print("\nComparison video finished.")
    print(f"Video saved to: {video_path}")
    print("\nFinal states:")
    print(
        f"no_recovery | recovered={no_recovery_final.recovered_once} | "
        f"success={no_recovery_final.task_success} | "
        f"d_goal={no_recovery_final.distance_to_goal:.3f}"
    )
    print(
        f"ppo_recovery | recovered={ppo_final.recovered_once} | "
        f"success={ppo_final.task_success} | "
        f"d_goal={ppo_final.distance_to_goal:.3f}"
    )


if __name__ == "__main__":
    main()