from omni.isaac.gym.vec_env import VecEnvBase
import torch

env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

task = RobotPuckTask(name="Cartpole")
env.set_task(task, backend="torch") 

from stable_baselines3 import DDPG
import optuna
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

def objective(trial: optuna.Trial) -> float:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    env._world.reset()
    obs, _ = env.reset()

    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical('buffer_size', [100000, 500000, 1000000])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    tau = trial.suggest_float('tau', 0.001, 0.01)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    sigma = trial.suggest_float('sigma', 0.01, 0.2)

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        "MlpPolicy", 
        env, 
        action_noise=action_noise, 
        batch_size=batch_size, 
        buffer_size=buffer_size, 
        learning_rate=learning_rate, 
        tau=tau,
        gamma=gamma,
        verbose=0,
        tensorboard_log="./ddpg_optimization/run1"
    )
    model.learn(total_timesteps=100000)

    reward_history = []
    max_iter = 2000

    env._world.reset()
    obs, _ = env.reset()
    while env._simulation_app.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)

        # "rewards" is a float the first iteration for some reason
        reward = rewards if isinstance(rewards, float) else rewards.item()

        reward_history.append(reward)
        num_iters = len(reward_history)
        if num_iters >= max_iter:
            break
    return torch.mean(torch.Tensor(reward_history)).item()
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(study.best_params)