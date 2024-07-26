import numpy as np
from pathlib import Path

import torch as th

from cathsim.rl.utils import (
    process_transitions,
    generate_experiment_paths,
    make_vec_env,
)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common import policies

from imitation.algorithms import bc


class CnnPolicy(policies.ActorCriticCnnPolicy):
    """A CNN policy for behavioral clonning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    model_path, log_path, eval_path = generate_experiment_paths("trial_3")

    rng = np.random.default_rng(0)
    env = make_vec_env(wrapper_kwargs=dict(time_limit=2000))

    trial_path = Path.cwd() / "rl" / "expert" / "trial_2"
    transitions = process_transitions(trial_path)

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: th.finfo(th.float32).max,
    )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        demonstrations=transitions,
        rng=rng,
    )

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=500)
    bc_trainer.save_policy(model_path / "bc_baseline.zip")

    rewards, lengths = evaluate_policy(
        bc_trainer.policy,
        env,
        n_eval_episodes=30,
        return_episode_rewards=True,
    )

    print(f"Reward after training: {np.mean(rewards)}")
    print(f"Lengths: {np.mean(lengths)}")
