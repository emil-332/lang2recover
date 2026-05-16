from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
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


RecoveryStrategy = Literal[
    "no_recovery",
    "scripted_recovery",
    "ppo_recovery",
]


@dataclass(frozen=True)
class EpisodeResult:
    strategy: str
    seed: int
    reward_mode: str
    success: bool
    recovered: bool
    recovery_required: bool
    steps_total: int
    steps_to_recover: int | None
    steps_to_success: int | None
    final_distance_to_goal: float
    final_distance_to_recovery: float


@dataclass(frozen=True)
class StrategySummary:
    strategy: str
    episodes: int
    success_rate: float
    recovery_rate: float
    mean_steps_to_recover: float | None
    mean_steps_to_success: float | None
    mean_final_distance_to_goal: float
    mean_final_distance_to_recovery: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--knock-step", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--scripted-recovery-steps", type=int, default=8)
    parser.add_argument("--reward-mode", type=str, default="language_generated")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["no_recovery", "scripted_recovery", "ppo_recovery"],
        choices=["no_recovery", "scripted_recovery", "ppo_recovery"],
    )
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


def move_cube_by_normal_policy(
    env,
    normal_policy: CubeLevelNormalPolicy,
) -> None:
    state = read_pushcube_state(env)
    normal_action = normal_policy.act(state)

    if not normal_policy.is_task_complete(state):
        set_cube_xy(env, state.cube_xy + normal_action.delta_xy)


def run_episode(
    strategy: RecoveryStrategy,
    seed: int,
    reward_mode: str,
    model: PPO | None,
    knock_step: int,
    max_steps: int,
    scripted_recovery_steps: int,
) -> EpisodeResult:
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

    already_knocked = False
    recovery_active = False
    recovered = False
    recovery_required = False
    task_success = False

    steps_to_recover: int | None = None
    steps_to_success: int | None = None

    scripted_start_xy: np.ndarray | None = None
    scripted_step_count = 0

    for step in range(max_steps):
        current_state = read_pushcube_state(env)
        decision = detector.classify(current_state)

        if step == knock_step and not already_knocked:
            knock_cube_by_xy_offset(env, offset_xy=(0.0, -0.18))
            already_knocked = True

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

            if decision.status == RecoveryStatus.NEEDS_RECOVERY:
                recovery_required = True
                recovery_active = True

                if strategy == "scripted_recovery":
                    scripted_start_xy = current_state.cube_xy.copy()
                    scripted_step_count = 0

        if already_knocked and not recovered:
            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

            if decision.status == RecoveryStatus.NEEDS_RECOVERY:
                recovery_required = True
                recovery_active = True

        if recovery_active and not recovered:
            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

            if strategy == "no_recovery":
                # The normal policy is assumed to be unreliable outside its
                # expected state distribution, so without recovery we stall.
                pass

            elif strategy == "scripted_recovery":
                if scripted_start_xy is None:
                    scripted_start_xy = current_state.cube_xy.copy()

                scripted_step_count += 1
                alpha = min(1.0, scripted_step_count / scripted_recovery_steps)

                new_cube_xy = (
                    (1.0 - alpha) * scripted_start_xy
                    + alpha * detector.recovery_center_xy
                )

                set_cube_xy(env, new_cube_xy)

            elif strategy == "ppo_recovery":
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

                set_cube_xy(env, current_state.cube_xy + recovery_action.delta_xy)

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            current_state = read_pushcube_state(env)
            decision = detector.classify(current_state)

            if decision.status == RecoveryStatus.RECOVERED:
                recovered = True
                recovery_active = False
                steps_to_recover = step

        elif not task_success:
            # Before the knock, or after recovery, the normal policy is allowed
            # to continue toward the task goal.
            move_cube_by_normal_policy(env, normal_policy)

        current_state = read_pushcube_state(env)

        if normal_policy.is_task_complete(current_state):
            task_success = True
            steps_to_success = step
            break

    final_state = read_pushcube_state(env)
    final_distance_to_goal = float(
        np.linalg.norm(final_state.cube_xy - final_state.goal_xy)
    )
    final_distance_to_recovery = float(
        np.linalg.norm(final_state.cube_xy - detector.recovery_center_xy)
    )

    env.close()

    return EpisodeResult(
        strategy=strategy,
        seed=seed,
        reward_mode=reward_mode,
        success=task_success,
        recovered=recovered,
        recovery_required=recovery_required,
        steps_total=step + 1,
        steps_to_recover=steps_to_recover,
        steps_to_success=steps_to_success,
        final_distance_to_goal=final_distance_to_goal,
        final_distance_to_recovery=final_distance_to_recovery,
    )


def summarize_results(results: list[EpisodeResult]) -> list[StrategySummary]:
    summaries: list[StrategySummary] = []

    strategies = sorted({result.strategy for result in results})

    for strategy in strategies:
        subset = [result for result in results if result.strategy == strategy]

        success_rate = float(np.mean([result.success for result in subset]))
        recovery_rate = float(np.mean([result.recovered for result in subset]))

        recover_steps = [
            result.steps_to_recover
            for result in subset
            if result.steps_to_recover is not None
        ]

        success_steps = [
            result.steps_to_success
            for result in subset
            if result.steps_to_success is not None
        ]

        mean_steps_to_recover = (
            float(np.mean(recover_steps)) if len(recover_steps) > 0 else None
        )
        mean_steps_to_success = (
            float(np.mean(success_steps)) if len(success_steps) > 0 else None
        )

        summaries.append(
            StrategySummary(
                strategy=strategy,
                episodes=len(subset),
                success_rate=success_rate,
                recovery_rate=recovery_rate,
                mean_steps_to_recover=mean_steps_to_recover,
                mean_steps_to_success=mean_steps_to_success,
                mean_final_distance_to_goal=float(
                    np.mean([result.final_distance_to_goal for result in subset])
                ),
                mean_final_distance_to_recovery=float(
                    np.mean(
                        [result.final_distance_to_recovery for result in subset]
                    )
                ),
            )
        )

    return summaries


def write_results_csv(path: Path, results: list[EpisodeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(results[0]).keys()),
        )
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))


def write_summary_csv(path: Path, summaries: list[StrategySummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(summaries[0]).keys()),
        )
        writer.writeheader()

        for summary in summaries:
            writer.writerow(asdict(summary))


def plot_success_rate(path: Path, summaries: list[StrategySummary]) -> None:
    strategies = [summary.strategy for summary in summaries]
    success_rates = [summary.success_rate for summary in summaries]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.bar(strategies, success_rates)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("Task success rate by recovery strategy")
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mean_steps(path: Path, summaries: list[StrategySummary]) -> None:
    strategies = [summary.strategy for summary in summaries]
    mean_steps = [
        np.nan
        if summary.mean_steps_to_success is None
        else summary.mean_steps_to_success
        for summary in summaries
    ]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.bar(strategies, mean_steps)
    ax.set_ylabel("Mean steps to task success")
    ax.set_title("Mean task-completion steps by recovery strategy")
    ax.grid(axis="y")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_summary(summaries: list[StrategySummary]) -> None:
    print("\nEvaluation summary:")
    print(
        f"{'strategy':>20} | "
        f"{'episodes':>8} | "
        f"{'success':>8} | "
        f"{'recovery':>8} | "
        f"{'steps_success':>13} | "
        f"{'final_d_goal':>12}"
    )
    print("-" * 82)

    for summary in summaries:
        steps_success = (
            "n/a"
            if summary.mean_steps_to_success is None
            else f"{summary.mean_steps_to_success:.2f}"
        )

        print(
            f"{summary.strategy:>20} | "
            f"{summary.episodes:8d} | "
            f"{summary.success_rate:8.2f} | "
            f"{summary.recovery_rate:8.2f} | "
            f"{steps_success:>13} | "
            f"{summary.mean_final_distance_to_goal:12.3f}"
        )


def main() -> None:
    args = parse_args()

    model: PPO | None = None

    if "ppo_recovery" in args.strategies:
        model_path = resolve_model_path(args.reward_mode)
        model = PPO.load(model_path)
        print(f"Loaded PPO model from: {model_path}")

    output_dir = Path("results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[EpisodeResult] = []

    print("\nRunning recovery-strategy evaluation.")
    print(f"Episodes per strategy: {args.episodes}")
    print(f"Strategies: {args.strategies}")
    print(f"Reward mode: {args.reward_mode}")

    for strategy in args.strategies:
        for episode_idx in range(args.episodes):
            seed = args.seed_start + episode_idx

            result = run_episode(
                strategy=strategy,
                seed=seed,
                reward_mode=args.reward_mode,
                model=model,
                knock_step=args.knock_step,
                max_steps=args.max_steps,
                scripted_recovery_steps=args.scripted_recovery_steps,
            )

            results.append(result)

            print(
                f"strategy={strategy:>18} | "
                f"seed={seed:04d} | "
                f"success={result.success} | "
                f"recovered={result.recovered} | "
                f"steps={result.steps_total:03d} | "
                f"final_d_goal={result.final_distance_to_goal:.3f}"
            )

    summaries = summarize_results(results)

    results_path = output_dir / "recovery_strategy_results.csv"
    summary_path = output_dir / "recovery_strategy_summary.csv"
    success_plot_path = output_dir / "success_rate_by_strategy.png"
    steps_plot_path = output_dir / "mean_steps_by_strategy.png"

    write_results_csv(results_path, results)
    write_summary_csv(summary_path, summaries)
    plot_success_rate(success_plot_path, summaries)
    plot_mean_steps(steps_plot_path, summaries)

    print_summary(summaries)

    print("\nSaved evaluation artifacts:")
    print(f"Per-episode results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Success plot: {success_plot_path}")
    print(f"Steps plot: {steps_plot_path}")


if __name__ == "__main__":
    main()