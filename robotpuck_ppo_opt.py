from omni.isaac.gym.vec_env import VecEnvBase
import torch

env = VecEnvBase(headless=True)

from robotpuck_task import RobotPuckTask

task = RobotPuckTask(name="Cartpole")
env.set_task(task, backend="torch") 

from stable_baselines3 import PPO
import optuna

def objective(trial: optuna.Trial) -> float:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    env._world.reset()
    obs, _ = env.reset()

    clip_range = trial.suggest_float('clip_range', 0.1, 0.4, log=True)
    n_epochs = trial.suggest_int('n_epochs', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.8, 0.9999)
    ent_coef = trial.suggest_float('ent_coef', 0.00001, 0.1, log=True)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0, log=True)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 5)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1000,
        clip_range=clip_range,
        batch_size=1000,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        device="cuda:0",
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=0,
        tensorboard_log="./ppo_optimization/run4"
    )
    model.learn(total_timesteps=200000)

    reward_history = []
    max_iter = 2000

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
    return torch.mean(torch.Tensor(reward_history)).item()
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)