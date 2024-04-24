from omni.isaac.gym.vec_env import VecEnvBase
import torch

env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

task = RobotPuckTask(name="Cartpole")
env.set_task(task, backend="torch") 

from stable_baselines3 import PPO

# create agent from stable baselines
model = PPO(
    "MlpPolicy",
    env,
    n_steps=1000,
    clip_range=0.2,
    batch_size=1000,
    n_epochs=5,
    learning_rate=0.005405026167593133,
    gamma=0.999,
    device="cuda:0",
    ent_coef=4.33542408133457e-05,
    vf_coef=0.0020737398594318467,
    max_grad_norm=0.9,
    verbose=1,
    tensorboard_log="./robotpuck_tensorboard"
)
model.learn(total_timesteps=1000000)
model.save("joint_obs_test")