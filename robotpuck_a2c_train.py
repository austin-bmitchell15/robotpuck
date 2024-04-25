from omni.isaac.gym.vec_env import VecEnvBase
import torch

# Initialize the environment
env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

# Set the task for the environment
task = RobotPuckTask(name="RobotPuck")
env.set_task(task, backend="torch")

from stable_baselines3 import A2C

# A2C model configuration
model = A2C(
    "MlpPolicy",
    env,
    learning_rate=0.000568693678278647,
    n_steps=100,
    gamma=0.9743440077383112,
    gae_lambda=0.9213666920087923,
    ent_coef=0.015497025538787557,
    vf_coef=0.7572941832896481,
    verbose=1
)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("a2c_policy")
