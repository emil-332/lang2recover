from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from lang2recover.envs.recovery_2d_env import Recovery2DEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path("checkpoints")
    log_dir = Path("results/ppo_recovery_2d")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(Recovery2DEnv(), filename=str(log_dir / "monitor.csv"))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=0.01,
        tensorboard_log=str(log_dir),
    )

    print("\nTraining PPO recovery policy.")
    print(f"Timesteps: {args.timesteps}")
    print("Environment: Recovery2DEnv")
    print("Goal: move displaced cube back into recovery zone.\n")

    model.learn(total_timesteps=args.timesteps)

    model_path = checkpoint_dir / "recovery_ppo_2d.zip"
    model.save(model_path)

    env.close()

    print(f"\nSaved PPO recovery policy to: {model_path}")


if __name__ == "__main__":
    main()