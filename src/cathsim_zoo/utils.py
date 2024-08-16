import os
from pathlib import Path

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}

RESULTS_PATH = Path.cwd() / "results"


def generate_experiment_paths(experiment_path: Path = None) -> tuple:
    if experiment_path.is_absolute() is False:
        experiment_path = RESULTS_PATH / experiment_path

    model_path = experiment_path / "models"
    eval_path = experiment_path / "eval"
    log_path = experiment_path / "logs"
    for directory_path in [experiment_path, model_path, log_path, eval_path]:
        directory_path.mkdir(parents=True, exist_ok=True)
    return model_path, log_path, eval_path


def load_sb3_model(path: Path, config_name: str = None) -> BaseAlgorithm:
    config = Config(config_name)
    algo_kwargs = config.get("algo_kwargs", {})

    model = SAC.load(
        path,
        custom_objects={"policy_kwargs": algo_kwargs.get("policy_kwargs", {})},
    )
    return model


def make_gym_env(
    n_envs: int = 1,
    env_kwargs: dict = dict(
        image_size=80,
        image_fn=lambda x: x,
    ),
) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    def _create_env() -> gym.Env:
        import cathsim

        env = gym.make("CathSim-v1", **env_kwargs)

        return env

    if n_envs > 1:
        envs = [_create_env for _ in range(n_envs)]
        env = SubprocVecEnv(envs)
    else:
        env = _create_env()

    env = Monitor(env) if n_envs == 1 else VecMonitor(env)

    return env
