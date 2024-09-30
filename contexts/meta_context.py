import collections.abc
from typing import Callable, get_type_hints, get_args, get_origin
from inspect import signature, Parameter
from dataclasses import dataclass
from functools import partial
from mujoco import mjx
from jax import numpy as jnp
import jax.tree_util
import equinox as eqx


@partial(jax.tree_util.register_dataclass,
         data_fields=['R', 'horizon', 'mx'],
         meta_fields=['lr', 'seed', 'nsteps', 'epochs', 'batch', 'vis', 'dt', 'path'])
@dataclass(frozen=True)
class Config:
    lr: float
    seed: int
    nsteps: int
    epochs: int
    batch: int
    vis: int
    dt: float
    path: str
    R: jnp.ndarray
    mx: mjx.Model

class Callbacks:
    def __init__(
            self,
            run_cost: Callable[[jnp.ndarray], jnp.ndarray],
            terminal_cost: Callable[[jnp.ndarray], jnp.ndarray],
            control_cost: Callable[[jnp.ndarray], jnp.ndarray],
            init_gen: Callable[[int, jnp.ndarray], jnp.ndarray],
            state_encoder: Callable[[jnp.ndarray], jnp.ndarray],
            state_decoder: Callable[[jnp.ndarray], jnp.ndarray],
            gen_network: Callable[[], eqx.Module],
            controller: Callable[[jnp.ndarray, jnp.ndarray, eqx.Module, Config, mjx.Model, mjx.Data], jnp.ndarray]
    ):
        self.run_cost = run_cost
        self.terminal_cost = terminal_cost
        self.control_cost = control_cost
        self.init_gen = init_gen
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.gen_network = gen_network
        self.controller = controller
        self._validate_callbacks()

    def _validate_callbacks(self):
        annotations = get_type_hints(self.__init__)
        for attr_name, expected_type in annotations.items():
            if attr_name == 'return':
                continue  # Skip return type

            func = getattr(self, attr_name)
            if get_origin(expected_type) is not collections.abc.Callable:
                raise TypeError(
                    f"Expected type of attribute '{attr_name}' is not a callable type."
                )

            expected_args_types, expected_return_type = get_args(expected_type)

            func_signature = signature(func)
            func_params = list(func_signature.parameters.values())

            # Check number of parameters
            if len(func_params) != len(expected_args_types):
                raise TypeError(
                    f"Function '{attr_name}' expects {len(expected_args_types)} parameters, "
                    f"but received {len(func_params)}."
                )

            # Check parameter types
            for param, expected_arg_type in zip(func_params, expected_args_types):
                # Handle cases where parameter has no annotation
                if param.annotation is Parameter.empty:
                    raise TypeError(
                        f"Parameter '{param.name}' in function '{attr_name}' lacks a type annotation."
                    )
                if param.annotation != expected_arg_type:
                    raise TypeError(
                        f"Parameter '{param.name}' in function '{attr_name}' has type {param.annotation}, "
                        f"expected {expected_arg_type}."
                    )

            # Check return type
            if func_signature.return_annotation is Parameter.empty:
                raise TypeError(
                    f"Function '{attr_name}' lacks a return type annotation."
                )
            if func_signature.return_annotation != expected_return_type:
                raise TypeError(
                    f"Function '{attr_name}' has return type {func_signature.return_annotation}, "
                    f"expected {expected_return_type}."
                )


class Context:
    def __init__(self, cfg: Config, cbs: Callbacks):
        self.cfg = cfg
        self.cbs = cbs
        assert jnp.all(jnp.linalg.eigh(self.cfg.R)[0] > 0), (
            "R should be positive definite."
        )
