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

    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 0.01, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
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
        tensorboard_log="./hyperparam_optimization"
    )
    model.learn(total_timesteps=50000)

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
    return torch.sum(torch.Tensor(reward_history)).item()
    
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(study.best_params)


# # create agent from stable baselines
# model = PPO(
#     "MlpPolicy",
#     env,
#     n_steps=1000,
#     clip_range=0.1,
#     batch_size=1000,
#     n_epochs=20,
#     learning_rate=0.001,
#     gamma=0.99,
#     device="cuda:0",
#     ent_coef=0.0,
#     vf_coef=0.5,
#     max_grad_norm=1.0,
#     verbose=1,
#     tensorboard_log="./robotpuck_tensorboard"
# )
# model.learn(total_timesteps=100000)
# model.save("joint_obs_test")