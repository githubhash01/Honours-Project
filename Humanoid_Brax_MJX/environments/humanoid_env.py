from etils import epath
from brax.envs.base import PipelineEnv, State
import mujoco 
from brax.io import mjcf
import jax 
from jax import numpy as jp 
from mujoco import mjx 

# TODO - replace this with local path to model
HUMANOID_ROOT_PATH = epath.Path(epath.resource_path('mujoco')) / 'mjx/test_data/humanoid'

class Humanoid(PipelineEnv):

  """
  __init__ 

  - initialises humanoid environment 
  - loads MuJoCo model (humanoid.xml)
  - configures physics simulation parameters 
  - defines reward structure, termination conditions, observation settings
  """
  def __init__(
      self,
      forward_reward_weight=1.25,
      ctrl_cost_weight=0.1,
      healthy_reward=5.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(1.0, 2.0),
      reset_noise_scale=1e-2,
      exclude_current_positions_from_observation=True,
      **kwargs,
  ):
#
    mj_model = mujoco.MjModel.from_xml_path(
        (HUMANOID_ROOT_PATH / 'humanoid.xml').as_posix()) # reads humanoid configuration file TODO - make this use local xml
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG # conjugate gradient method constrained solver 
    mj_model.opt.iterations = 6 # 6 solver iterations 
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model) # converst MuJoCo XML into Brax-compatible model 

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx' # uses mjx as the backend 

    super().__init__(sys, **kwargs)

    self._forward_reward_weight = forward_reward_weight
    self._ctrl_cost_weight = ctrl_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

  """
  reset

  - rests humanoid to initial state 
  - introduces random noise to make learning more generalisable

  """
  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel) # initialise the humanoid state in physics engine

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'forward_reward': zero,
        'reward_linvel': zero,
        'reward_quadctrl': zero,
        'reward_alive': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
    }
    return State(pipeline_state=data, obs=obs, reward=reward, done=done, metrics=metrics, info={})

  """
  step

  - applies action (joint torques) to the humanoid 
  - simulates the physics, computes reward and checks for termination
  """
  def step(self, state: State, action: jp.ndarray) -> State:
    
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state # get the current state 
    data = self.pipeline_step(data0, action) # advance physics by one step 

    # compute how much the humanoid moved forward 
    com_before = data0.subtree_com[1] 
    com_after = data.subtree_com[1]
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0]

    # make sure the humanoid has not fallen down 
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(data.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(data.q[2] > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(data, action)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        forward_reward=forward_reward,
        reward_linvel=forward_reward,
        reward_quadctrl=-ctrl_cost,
        reward_alive=healthy_reward,
        x_position=com_after[0],
        y_position=com_after[1],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
    )

    # having calculated the new humanoid state, update the state vector 
    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )
  
  """
  _get_obs

  - constructs the state observation for the RL agent 
  """

  def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:

    """Observes humanoid body position, velocities, and angles."""
    position = data.qpos
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        data.qvel,
        data.cinert[1:].ravel(),
        data.cvel[1:].ravel(),
        data.qfrc_actuator,
    ])
