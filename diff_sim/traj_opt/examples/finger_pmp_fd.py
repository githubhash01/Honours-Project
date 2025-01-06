import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
from diff_sim.traj_opt.pmp_fd import PMP, make_loss_fn, make_step_fn

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("../../xmls/finger_mjx.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([-.8, 0, -.8])
    Nsteps, nu = 10, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + 0.01 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    loss_fn = make_loss_fn(mx, qpos_init, set_control, running_cost, terminal_cost)
    optimizer = PMP(loss=loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.2, max_iter=50)

    from diff_sim.utils.mj import visualise_traj_generic
    from diff_sim.traj_opt.pmp_fd import simulate_trajectory
    import mujoco

    d = mujoco.MjData(model)
    step_func = make_step_fn(mx, dx, set_control)
    x, cost = simulate_trajectory(mx, qpos_init, step_func, running_cost, terminal_cost, optimal_U)
    visualise_traj_generic(jnp.expand_dims(x, axis=0), d, model)
