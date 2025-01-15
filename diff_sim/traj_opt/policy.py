import jax
from jaxtyping import PyTree
import jax.numpy as jnp
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable, Optional, Set
import equinox
from jax.flatten_util import ravel_pytree
import numpy as np
from jax._src.util import unzip2
 
config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)
  
# -------------------------------------------------------------
# Helpers for attribute names
# -------------------------------------------------------------
class GetAttrKey:
    def __init__(self, name):
        self.name = name
 
    def __repr__(self):
        return f"GetAttrKey(name='{self.name}')"
 
 
def filter_state_data(dx: mjx.Data):
    """
    Select a subset of the mjx.Data fields (qpos, qvel, qacc, sensordata, etc.)
    that you want to include in your derivative.
    """
    return (
        dx.qpos,
        dx.qvel,
        dx.qacc,
        dx.sensordata,
        dx.mocap_pos,
        dx.mocap_quat
    ) 
 
# -------------------------------------------------------------
# Finite-difference cache
# -------------------------------------------------------------
@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    sensitivity_mask: jnp.ndarray
    inner_idx: jnp.ndarray
    dx_size: int
    num_u_dims: int
    eps: float = 1e-6
 
 
def build_fd_cache(
    dx_ref: mjx.Data,
    u_ref: jnp.ndarray,
    target_fields: Optional[Set[str]] = None,
    eps: float = 1e-6
) -> FDCache:
    """
    Build a cache containing:
      - Flatten/unflatten for dx_ref
      - The mask for relevant FD indices (e.g. qpos, qvel, ctrl)
      - The shape info for control
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel", "ctrl"}
 
    # Flatten dx
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]
 
    # Flatten control
    u_ref_flat = u_ref.ravel()
    num_u_dims = u_ref_flat.shape[0]
 
    # Gather leaves for qpos, qvel, ctrl
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
    sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
    indices = tuple(np.cumsum(sizes))
 
    idx_target_state = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        # Check if any level in the path has a 'name' that is in target_fields
        name_matches = any(
            getattr(level, 'name', None) in target_fields
            for level in path
        )
        if name_matches:
            idx_target_state.append(i)
 
    def leaf_index_range(leaf_idx):
        start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
        end = indices[leaf_idx]
        return np.arange(start, end)
 
    # Combine all relevant leaf sub-ranges
    inner_idx_list = []
    for i in idx_target_state:
        inner_idx_list.append(leaf_index_range(i))
    inner_idx = np.concatenate(inner_idx_list, axis=0)
    inner_idx = jnp.array(inner_idx, dtype=jnp.int32)
 
    # Build the sensitivity mask
    sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)
 
    return FDCache(
        unravel_dx = unravel_dx,
        sensitivity_mask = sensitivity_mask,
        inner_idx = inner_idx,
        dx_size = dx_size,
        num_u_dims = num_u_dims,
        eps = eps
    )
 
 
# -------------------------------------------------------------
# Step function with custom FD-based derivative
# -------------------------------------------------------------
def make_step_fn(
        mx,
        set_control_fn: Callable,
        fd_cache: FDCache
):
    """
    Create a custom_vjp step function that takes (dx, u) and returns dx_next.
    We do finite differences (FD) in the backward pass using the info in fd_cache.
    """
 
    @jax.custom_vjp
    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next
 
    def step_fn_fwd(dx, u):
        dx_next = step_fn(dx, u)
        return dx_next, (dx, u, dx_next)
 
    def step_fn_bwd(res, g):
        """
        FD-based backward pass. We approximate d(dx_next)/d(dx,u) and chain-rule with g.
        Uses the cached flatten/unflatten info in fd_cache.
        """
        dx_in, u_in, dx_out = res
 
        # Convert float0 leaves in 'g' to zeros
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf
            return jax.tree_map(fix_leaf, diff_tree, grad_tree)
 
        mapped_g = map_g_to_dinput(dx_in, g)
        g_array, _ = ravel_pytree(mapped_g)
 
        # Flatten dx_in and dx_out just once
        dx_array, _ = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)
        u_in_flat = u_in.ravel()
 
        # Grab cached info
        unravel_dx = fd_cache.unravel_dx
        sensitivity_mask = fd_cache.sensitivity_mask
        inner_idx = fd_cache.inner_idx
        num_u_dims = fd_cache.num_u_dims
        eps = fd_cache.eps
 
        # =====================================================
        # =============== FD wrt control (u) ==================
        # =====================================================
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps
 
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))  # shape = (num_u_dims, dx_dim)
 
        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps
 
        # Only do FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)
 
        # Scatter those rows back to a full (dx_dim, dx_dim) matrix
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)
 
        dx_dim = dx_array.size
        Jx_array = scatter_rows(Jx_rows, inner_idx, (dx_dim, dx_dim))
 
        # =====================================================
        # ================== Combine with g ====================
        # =====================================================
        d_u_flat = Ju_array @ g_array     # shape = (num_u_dims,)
        d_x_flat = Jx_array @ g_array     # shape = (dx_dim,)
 
        d_x = unravel_dx(d_x_flat)
        d_u = d_u_flat.reshape(u_in.shape)
 
        return (d_x, d_u)
 
    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn
 
 
# -------------------------------------------------------------
# Simulate a trajectory
# -------------------------------------------------------------
@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, running_cost_fn, terminal_cost_fn, step_fn, params, static, length):
    """
    (Unchanged, except that step_fn is now the new custom FD step_fn)
    Simulate a trajectory with a policy = model(x, t).
    """
    model = equinox.combine(params, static)
 
    def scan_step_fn(dx, _):
        x = jnp.concatenate([dx.qpos, dx.qvel])
        u = model(x, dx.time)
        dx = step_fn(dx, u)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)
 
    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(scan_step_fn, dx0, length=length)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost
 
 
# -------------------------------------------------------------
# Build the loss
# -------------------------------------------------------------
def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, length):
    """
    Create a loss function that takes 'params' (your policy parameters) + 'static' and
    returns total trajectory cost. We build the FDCache once here.
    """
 
    # Build a reference data for FD shape
    dx_ref = mjx.make_data(mx)
    # Build a reference control for FD shape (adjust if your control dimension differs)
    u_ref = jnp.zeros((mx.nu,))  # shape = (nu,)
 
    # Build the FD cache once
    fd_cache = build_fd_cache(
        dx_ref,
        u_ref,
        target_fields={'qpos', 'qvel', 'ctrl'},  # or whichever fields you need
        eps=1e-6
    )
 
    # Create the step function with FD-based custom VJP
    step_fn = make_step_fn(mx, set_control_fn, fd_cache)
 
    def loss(params, static):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            running_cost_fn, terminal_cost_fn, step_fn,
            params, static, length
        )
        return total_cost
 
    return loss
 
 
# -------------------------------------------------------------
# The Policy class for gradient-based optimization
# -------------------------------------------------------------
@dataclass
class Policy:
    loss: Callable[[PyTree, PyTree], float]
    grad_loss: Callable[[PyTree, PyTree], jnp.ndarray]
 
    def solve(self, model: equinox.Module, optim, state, max_iter=100):
        """
        (Unchanged)
        Generic gradient descent loop on your policy parameters.
        """
        opt_model = None
        for i in range(max_iter):
            params, static = equinox.partition(model, equinox.is_array)
            g = self.grad_loss(params, static)
            f_val = self.loss(params, static)
            updates, state = optim.update(g, state, model)
            model = equinox.apply_updates(model, updates)
            opt_model = model
            print(f"Iteration {i}: cost={f_val}")
        return model
 