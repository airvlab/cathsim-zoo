import os
from pathlib import Path

import cv2
import gymnasium as gym
import wandb
from cathsim.wrappers import MultiInputImageWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecVideoRecorder

id = 4


def image_fn(image):
    return image


def make_gym_env(
    n_envs: int = 1,
    env_kwargs: dict = dict(
        image_size=128,
        image_fn=lambda x: x,
        render_mode="rgb_array",
    ),
) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    def _create_env() -> gym.Env:
        import cathsim

        env = gym.make("CathSim-v1", **env_kwargs)
        env = MultiInputImageWrapper(
            env,
            grayscale=True,
            image_key="pixels",
            keep_dim=True,
        )

        return env

    envs = [_create_env for _ in range(n_envs)]
    env = SubprocVecEnv(envs)
    env = VecMonitor(env)
    env = VecVideoRecorder(
        env,
        video_folder=f"results/videos/{id}",
        record_video_trigger=lambda x: x % 10_000 == 0,
        video_length=1000,
    )

    return env


def train(
    n_envs: int = 1,
    n_timesteps: int = 600_000,
) -> None:
    env = make_gym_env(n_envs=n_envs)

    model = SAC(
        "MultiInputPolicy",
        env=env,
        verbose=1,
        tensorboard_log=f"results/runs/{id}",
        gradient_steps=-1,
        seed=0,
    )

    model.learn(
        total_timesteps=n_timesteps,
        progress_bar=True,
    )


if __name__ == "__main__":
    train(
        n_envs=os.cpu_count() // 2,
        n_timesteps=600_000,
    )
