import jax
import jax.numpy as jnp
from mujoco import mjx
import equinox as eqx
from diff_sim.context.meta_context import Context
from diff_sim.nn.base_nn import Network

@eqx.filter_jit
def controlled_simulate(x_inits:jnp.ndarray, ctx: Context, net: Network, key: jnp.ndarray):
    mx = ctx.cfg.mx

    def cost_fn(mx:mjx.Model , dx:mjx.Data):
        ucost = ctx.cbs.control_cost(mx,dx) * ctx.cfg.dt
        xcst = ctx.cbs.run_cost(mx,dx) * ctx.cfg.dt
        return jnp.array([ucost + xcst])

    def set_init(x):
        dx = mjx.make_data(mx)
        # TODO: Decode x_init here.
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    # TODO: append a flag with x that signifies wether you should terminate or not
    def step(carry, _):
        dx, key = carry
        key, subkey = jax.random.split(key)
        dx, u = ctx.cbs.controller(net, mx, dx, subkey)
        cost = cost_fn(mx, dx)
        dx = mjx.step(mx, dx) # Dynamics function
        x = ctx.cbs.state_encoder(mx,dx)
        t = jnp.expand_dims(dx.time, axis=0)
        return (dx, key), jnp.concatenate([x, dx.ctrl, cost, t], axis=0)

    @jax.vmap
    def rollout(x_init):
        dx = set_init(x_init)
        (dx,_), res = jax.lax.scan(step, (dx, key), None, length=ctx.cfg.nsteps-1)
        x, u, costs, ts = res[...,:-mx.nu-2], res[...,-mx.nu-2:-2], res[...,-2], res[...,-1]
        x = jnp.concatenate([x_init.reshape(1,-1), x], axis=0)
        t = jnp.concatenate([jnp.array([ctx.cfg.dt]), ts], axis=0)
        tcost = ctx.cbs.terminal_cost(mx,dx) # Terminal cost
        costs = jnp.concatenate([costs, tcost.reshape(-1)], axis=0)
        return x, u, costs, t #return dx as well return termination indicies as well

    return rollout(x_inits)
