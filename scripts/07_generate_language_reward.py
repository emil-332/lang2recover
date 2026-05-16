from __future__ import annotations

from pathlib import Path

from lang2recover.rewards.language_reward_codegen import generate_reward_from_spec


def main() -> None:
    spec_path = Path("reward_specs/cube_recovery.yaml")
    output_dir = Path("generated_rewards")

    paths = generate_reward_from_spec(
        spec_path=spec_path,
        output_dir=output_dir,
    )

    print("Generated language-shaped reward artifacts:")
    print(f"Prompt: {paths.prompt_path}")
    print(f"Reward module: {paths.reward_module_path}")


if __name__ == "__main__":
    main()