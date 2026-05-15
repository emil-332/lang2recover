import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


def _done(x) -> bool:
    try:
        return bool(x.item())
    except AttributeError:
        return bool(x)


def main() -> None:
    env = gym.make(
        "PushCube-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    obs, info = env.reset(seed=0)

    print("Environment created successfully.")
    print("Action space:", env.action_space)

    for step in range(80):
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0:
            print(f"step={step:03d} | reward={reward}")

        if _done(terminated) or _done(truncated):
            break

    env.close()
    print("Smoke test finished without rendering.")


if __name__ == "__main__":
    main()