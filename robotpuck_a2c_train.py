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
    n_steps= 20, 
    learning_rate= 0.001570376198958359, 
    gamma= 0.9154785103133573, 
    gae_lambda= 0.9644010054106701, 
    ent_coef= 0.09093196310755056, 
    vf_coef= 0.43795677412063205,
    verbose=1,
    tensorboard_log="./a2c_training/"
)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("a2c_policy")
