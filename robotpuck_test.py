from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=False)

# create task and register task
from robotpuck_task import RobotPuckTask
task = RobotPuckTask(name="Puck")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import PPO

# Run inference on the trained policy
model = PPO.load("3000_force_new_reward_train")
env._world.reset()
obs, _ = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

env.close()