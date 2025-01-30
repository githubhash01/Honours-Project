from brax.io import model
import jax 
import cv2

def save_video(frames, filename, fps):
    """
    Saves frames as a video file using OpenCV.
    """
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure fps is treated as a float
    fps = float(fps)
    
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()

def visualise_policy_rollout(model_path, make_inference_fn, eval_env, rollout_path, n_steps=500, render_every=2):
    """
    Loads a trained policy and visualizes the rollout in the given environment.
    """
    print("Saving Visualisation")
    params = model.load_params(model_path)
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    
    # Initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    frames = eval_env.render(rollout[::render_every], camera='side')
    
    fps = 1.0 / eval_env.dt / render_every
    save_video(frames, rollout_path, fps=fps)
    print(f"Visualization saved to {rollout_path}.")
