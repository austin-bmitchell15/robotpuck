from omni.isaac.gym.vec_env import VecEnvBase
env = VecEnvBase(headless=True)

import matplotlib.pyplot as plt

# create task and register task
from robotpuck_task import RobotPuckTask
task = RobotPuckTask(name="Puck")
env.set_task(task, backend="torch")

# import stable baselines
from stable_baselines3 import PPO

reward_history = []
max_iter = 2000

# Run inference on the trained policy
model = PPO.load("joint_obs_test")
env._world.reset()
obs, _ = env.reset()
while env._simulation_app.is_running():
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)

    # "rewards" is a float the first iteration for some reason
    reward = rewards if isinstance(rewards, float) else rewards.item()

    reward_history.append(reward)
    num_iters = len(reward_history)
    if num_iters >= max_iter:
        break

plt.plot(reward_history)
plt.show()

env.close()
