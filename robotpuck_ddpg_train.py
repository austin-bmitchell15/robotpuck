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

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy", 
    env, 
    action_noise=action_noise, 
    batch_size= 512, 
    buffer_size= 100000, 
    learning_rate= 0.002192963766359842, 
    tau= 0.0015661843317282323, 
    gamma= 0.9988993766824212,
    verbose=1
)

model.learn(total_timesteps=100000)
model.save("ddpg_policy")