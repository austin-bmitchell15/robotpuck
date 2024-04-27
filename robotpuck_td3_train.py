from omni.isaac.gym.vec_env import VecEnvBase
import torch

env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

task = RobotPuckTask(name="Cartpole")
env.set_task(task, backend="torch") 

from stable_baselines3 import TD3
#import optuna
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MlpPolicy", 
    env, 
    action_noise=action_noise, 
    verbose=1,
    tensorboard_log="./td3_optimization"
)

model.learn(total_timesteps=200000)
model.save("td3_policy")