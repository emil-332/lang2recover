from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from lang2recover.envs.recovery_2d_env import Recovery2DConfig, Recovery2DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="manual_dense",
        choices=["manual_dense", "language_generated"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def render_frame(
    cube_xy: np.ndarray,
    recovery_center_xy: np.ndarray,
    goal_xy: np.ndarray,
    step: int,
    action: np.ndarray | None,
    reward: float | None,
    distance_to_recovery: float,
    is_recovered: bool,
    reward_mode: str,
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
        label="Original task goal",
    )

    ax.add_patch(recovery_circle)
    ax.add_patch(unexpected_circle)
    ax.add_patch(goal_circle)

    ax.scatter([cube_xy[0]], [cube_xy[1]], s=160, label="Cube")
    ax.scatter(
        [recovery_center_xy[0]],
        [recovery_center_xy[1]],
        s=100,
        marker="x",
        label="Recovery center",
    )
    ax.scatter([goal_xy[0]], [goal_xy[1]], s=140, marker="*", label="Task goal")

    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        ax.arrow(
            cube_xy[0],
            cube_xy[1],
            action[0] * 0.04,
            action[1] * 0.04,
            head_width=0.01,
            length_includes_head=True,
        )

    span = 0.38
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    status = "RECOVERED" if is_recovered else "RECOVERING"
    reward_text = "n/a" if reward is None else f"{reward:.3f}"

    ax.set_title(
        f"Lang2Recover PPO Recovery | step={step} | {status}\n"
        f"reward_mode={reward_mode} | distance={distance_to_recovery:.3f} | reward={reward_text}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()

    model_path = Path(f"checkpoints/recovery_ppo_2d_{args.reward_mode}.zip")

    if not model_path.exists() and args.reward_mode == "manual_dense":
        model_path = Path("checkpoints/recovery_ppo_2d.zip")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. "
            f"Run scripts/04_train_recovery_ppo.py --reward-mode {args.reward_mode} first."
        )

    output_dir = Path(f"videos/05_ppo_recovery_policy/{args.reward_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "ppo_recovery_policy.mp4"

    env = Recovery2DEnv(config=Recovery2DConfig(reward_mode=args.reward_mode))
    model = PPO.load(model_path)

    obs, info = env.reset(seed=args.seed)

    frames = []

    cube_xy = info["cube_xy"]
    recovery_center_xy = info["recovery_center_xy"]
    goal_xy = info["goal_xy"]

    frames.append(
        render_frame(
            cube_xy=cube_xy,
            recovery_center_xy=recovery_center_xy,
            goal_xy=goal_xy,
            step=0,
            action=None,
            reward=None,
            distance_to_recovery=info["distance_to_recovery"],
            is_recovered=info["is_recovered"],
            reward_mode=args.reward_mode,
        )
    )

    total_reward = 0.0
    success = False

    for step in range(1, 80):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)

        cube_xy = info["cube_xy"]
        recovery_center_xy = info["recovery_center_xy"]
        goal_xy = info["goal_xy"]

        frames.append(
            render_frame(
                cube_xy=cube_xy,
                recovery_center_xy=recovery_center_xy,
                goal_xy=goal_xy,
                step=step,
                action=action,
                reward=float(reward),
                distance_to_recovery=info["distance_to_recovery"],
                is_recovered=info["is_recovered"],
                reward_mode=args.reward_mode,
            )
        )

        print(
            f"step={step:03d} | "
            f"action={np.round(action, 3)} | "
            f"reward={float(reward): .3f} | "
            f"distance={info['distance_to_recovery']:.3f} | "
            f"recovered={info['is_recovered']} | "
            f"reward_mode={args.reward_mode}"
        )

        if info["is_recovered"]:
            success = True

        if terminated or truncated:
            break

    env.close()

    imageio.mimsave(video_path, frames, fps=10)

    print("\nEvaluation finished.")
    print(f"Reward mode: {args.reward_mode}")
    print(f"Success: {success}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Video saved to: {video_path}")


if __name__ == "__main__":
    main()