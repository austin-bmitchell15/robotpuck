from omni.isaac.gym.vec_env import VecEnvBase
import torch

env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

task = RobotPuckTask(name="Cartpole")
env.set_task(task, backend="torch") 

from stable_baselines3 import A2C
import optuna

def objective(trial: optuna.Trial):
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    # Reset environment
    env._world.reset()
    obs, _ = env.reset()

    # Hyperparameters
    n_steps = trial.suggest_categorical('n_steps', [5, 10, 20, 50, 100])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.90, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.90, 0.99)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

    # Create A2C model
    model = A2C(
        "MlpPolicy", 
        env, 
        learning_rate=learning_rate, 
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=0,
        tensorboard_log="./a2c_optimization/run1"
    )
    model.learn(total_timesteps=100000)

    # Evaluate the model
    reward_history = []
    max_iter = 2000

    # Reset environment
    env._world.reset()
    obs, _ = env.reset()
    while env._simulation_app.is_running():
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = env.step(action)
        reward = rewards if isinstance(rewards, float) else rewards.item()
        reward_history.append(reward)
        if len(reward_history) >= max_iter:
            break

    return torch.mean(torch.Tensor(reward_history)).item()

# Optimize hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
print("Best hyperparameters:", study.best_params)
