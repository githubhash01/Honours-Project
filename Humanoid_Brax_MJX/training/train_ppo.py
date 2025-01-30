import functools 
from brax.training.agents.ppo import train as ppo
import matplotlib.pyplot as plt
from datetime import datetime

def train_ppo(env, training_plot_path):
    """
    Trains an agent using PPO 
    """
    train_fn = functools.partial(
        ppo.train, 
        num_timesteps=20_000_000,
        num_evals=5, 
        reward_scaling=0.1,
        episode_length=1000, 
        normalize_observations=True, 
        action_repeat=1,
        unroll_length=10, 
        num_minibatches=24, 
        num_updates_per_batch=8,
        discounting=0.97, 
        learning_rate=3e-4, 
        entropy_cost=1e-3, 
        num_envs=3072,
        batch_size=512, 
        seed=0)


    max_y, min_y = 13000, 0

    def progress(num_steps, metrics):
        x_data = []
        y_data = []
        ydataerr = []
        times = [datetime.now()]

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(
            x_data, y_data, yerr=ydataerr)

        # Save the plot with the current date in the filename
        plt.savefig(training_plot_path)

    print("Starting PPO training...")
    inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    
    print("Training complete.")
    return inference_fn, params
