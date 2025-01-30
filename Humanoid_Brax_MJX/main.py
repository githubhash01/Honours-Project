from brax import envs
from environments.humanoid_env import Humanoid
from training.train_ppo import train_ppo
from renderer.visualiser import visualise_policy_rollout
import jax 
from jax import numpy as jp 
import datetime
from brax.io import model 

def main():
    """
    Main function to train the humanoid using PPO.
    """
    # Step 0: Ensure JAX is using GPU
    print(jax.devices())

    # Step 1: Define the environment and set the state
    print("Registering environment")
    envs.register_environment('humanoid', Humanoid)
    env = envs.get_environment('humanoid')

    # Step 2: Train the policy 
    print("Training Humanoid")
    # Get the current date
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    training_plot_path = f"/home/hashim/Desktop/Humanoid_Brax_MJX/plots/training_plot_{current_date}.png"
    make_inference_fn, params = train_ppo(env, training_plot_path)
    # save the model 
    saved_model_path = f"/home/hashim/Desktop/Humanoid_Brax_MJX/saved_policies/humanoid_policy{current_date}"
    model.save_params(saved_model_path, params)

    # Step 3: Save the rollout
    rollout_path = f"/home/hashim/Desktop/Humanoid_Brax_MJX/rollouts/rollout{current_date}.mp4"
    visualise_policy_rollout(model_path=saved_model_path, make_inference_fn=make_inference_fn, eval_env=env, rollout_path=rollout_path)

if __name__=="__main__":
    main()