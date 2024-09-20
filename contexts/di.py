import jax
from jax import numpy as jnp
import equinox as eqx
from .meta_context import Config, Callbacks, Context
import os
import jax.debug
from mujoco import mjx

try:
    # This works when __file__ is defined (e.g., in regular scripts)
    base_path = os.path.dirname(__file__)
except NameError:
    # Fallback to current working directory (e.g., in interactive sessions)
    base_path = os.getcwd()

base_path = os.path.join(base_path, '../xmls')

class ValueFunc(eqx.Module):
    layers: list
    # act: lambda x: jax.nn.relu(x)
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

        # new_weight = jnp.array([[1.5,1.], [1.,1.5]])
        # new_biases = jnp.array([0.,0.])
        # where = lambda l: l.weight
        # self.layers[0] = eqx.tree_at(where, self.layers[0], new_weight)        

    def __call__(self, x, t):
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        # transformed_x = self.layers2[0](x)
        # return transformed_x @ x
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)

        # f = lambda x: jnp.einsum('...i,ij,...j->...', x, self.layers[0].weight, x)
        # v = f(x)
        # return v

        # PD Controller
        #     f = lambda x: jnp.einsum('...i,ij,...j->...', x, jnp.array([[1.3, 1],[1,1.3]]), x)
        #     v = f(x)
        #     return v
    
    @staticmethod
    def make_step(optim, model, state, loss, x, y):
        params, static = eqx.partition(model, eqx.is_array)
        loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, loss_value

    @staticmethod
    def loss_fn(params, static, x, y):
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
        y = y.reshape(-1, 1)
        return jnp.mean(jnp.square(pred - y))


def policy(x,t,net,cfg, mx, dx):
    return net(x,t)

def hjb(x,t,net,cfg, mx, dx):
    act_id = mx.actuator_trnid[:, 0]
    M = mjx.full_m(mx, dx)
    invM = jnp.linalg.inv(M)
    dvdx = jax.jacrev(net,0)(x, t)
    G = jnp.vstack([jnp.zeros_like(invM), invM])
    invR = jnp.linalg.inv(cfg.R)
    u = (-1/2 * invR @ G.T[act_id, :] @ dvdx.T).flatten()
    return u    


ctx = Context(cfg=Config(
    model_path=os.path.join(base_path, 'doubleintegrator.xml'),
    dims=[2, 64, 64, 1],
    lr=4.e-3,
    seed=0,
    nsteps=100,
    epochs=400,
    batch=50,
    vis=50,
    dt=0.01,
    R=jnp.array([[.01]]),
    horizon=jnp.arange(0, (100 + 1)*0.01, 0.01) + 0.01
    ),cbs=Callbacks(
        # run_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([10., .1])), x),
        # terminal_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([1., .01])), x),
        # terminal_cost = lambda x: 10*jnp.sum(jnp.abs(jnp.dot(jnp.diag(jnp.array([10., 0.1])), x.T).T), axis=-1),
        # control_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.array([[.1]]), x).at[..., -1].set(0),
        run_cost= lambda x: jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([10.,0.1])), x)),
        control_cost= lambda x: jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([0.01])), x)),
        terminal_cost= lambda x: 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([10.,0.1])), x)),
        init_gen= lambda batch, key: jnp.concatenate([
            jax.random.uniform(key, (batch, 1), minval=-1., maxval=1.),
            jax.random.uniform(key, (batch, 1), minval=-0.7, maxval=0.7)
        ], axis=1).squeeze(),
    state_encoder=lambda x: x,
    gen_network = lambda : ValueFunc([3, 64, 64, 1], jax.random.PRNGKey(0)),
    controller = hjb
    )
)

