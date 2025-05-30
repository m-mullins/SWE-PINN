from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional, Dict

from flax.training import train_state
from flax import jax_utils

import jax.numpy as jnp
from jax import lax, jit, grad, pmap, random, tree_map, jacfwd, jacrev, value_and_grad, debug
from jax.tree_util import tree_map, tree_reduce, tree_leaves

import optax

from jaxpi import archs
from jaxpi.utils import flatten_pytree


class TrainState(train_state.TrainState):
    weights: Dict   # For loss terms (IC, BC, PDE)
    momentum: float
    rba_weights: jnp.ndarray  # Collocation-point-specific RBA weights

    def apply_weights(self, weights, **kwargs):
        """Updates `weights` using running average  in return value.

        Returns:
          An updated instance of `self` with new weights updated by applying `running_average`,
          and additional attributes replaced as specified by `kwargs`.
        """

        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )
    
    def apply_rba_weights(self, rba_weights):
        """Updates RBA weights, called every iteration"""
        return self.replace(rba_weights=rba_weights)


def _create_arch(config):
    if config.arch_name == "Mlp":
        arch = archs.Mlp(**config)

    elif config.arch_name == "ResNet":
        arch = archs.ResNet(**config)

    elif config.arch_name == "ModifiedMlp":
        arch = archs.ModifiedMlp(**config)

    elif config.arch_name == "PIResNet":
        arch = archs.PIResNet(**config)

    elif config.arch_name == "PirateNet":
        arch = archs.PirateNet(**config)

    elif config.arch_name == "DeepONet":
        arch = archs.DeepONet(**config)

    else:
        raise NotImplementedError(f"Arch {config.arch_name} not supported yet!")

    return arch


def _create_optimizer(config):
    if config.optimizer == "Adam":
        lr = optax.exponential_decay(
            init_value=config.learning_rate,
            transition_steps=config.decay_steps,
            decay_rate=config.decay_rate,
            staircase=config.staircase,
        )

        if config.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0,
                end_value=config.learning_rate,
                transition_steps=config.warmup_steps,
            )

            lr = optax.join_schedules([warmup, lr], [config.warmup_steps])

        tx = optax.adam(
            learning_rate=lr, b1=config.beta1, b2=config.beta2, eps=config.eps
        )

    else:
        raise NotImplementedError(f"Optimizer {config.optimizer} not supported yet!")

    # Gradient accumulation
    if config.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.grad_accum_steps)

    return tx


def _create_train_state(config, params=None, weights=None, rba_weights=None):
    # Initialize network
    arch = _create_arch(config.arch)
    x = jnp.ones(config.input_dim)

    # Initialize optax optimizer
    tx = _create_optimizer(config.optim)

    if params is None:
        params = arch.init(random.PRNGKey(config.seed), x)

    if weights is None:
        weights = dict(config.weighting.init_weights)

    if rba_weights is None:
        rba_weights = jnp.ones(1)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=weights,
        rba_weights=rba_weights,
        momentum=config.weighting.momentum,
    )

    return jax_utils.replicate(state)


class PINN:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)

    def u_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def r_net(self, params, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def losses(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError("Subclasses should implement this!")

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, step, rba_weights, *args):
        # Compute losses
        losses, residuals = self.losses(params, batch, step, rba_weights, *args)
        # losses = self.losses(params, batch, step, *args)

        # Compute weighted loss
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        # Sum weighted losses
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss, residuals
    
    @partial(jit, static_argnums=(0,))
    def loss_grads(self, params, weights, batch, step, rba_weights, *args):
        # Define a closure that returns scalar loss + aux outputs
        def loss_fn(p):
            losses, residuals = self.losses(p, batch, step, rba_weights, *args)
            weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
            total_loss = tree_reduce(lambda x, y: x + y, weighted_losses)
            return total_loss, residuals  # Return both loss and residuals
        
        # Compute gradients while preserving residuals
        (loss, residuals), grads = value_and_grad(
            loss_fn, 
            has_aux=True  # Important for residual handling
        )(params)
        
        return loss, grads
    
    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def residuals(self, state, batch, *args):
        # Compute only the forward pass (total_loss and residuals)
        total_loss, residuals = self.loss(state.params, state.weights, batch, state.step, state.rba_weights)
        # Stop gradient computation on the residuals so that they’re treated as constants
        residuals = lax.stop_gradient(residuals)

        summed_residuals = jnp.sum(jnp.stack(residuals), axis=0)
        return summed_residuals

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, step, *args):
        if self.config.weighting.scheme == "grad_norm":
            # Wrap losses() to return only the first output (loss_dict), ignore residuals
            loss_fn = lambda p, b, s, rba: self.losses(p, b, s, rba)[0]

            # Compute the gradient of each loss w.r.t. the parameters
            grads = jacrev(loss_fn)(params, batch, step, self.state.rba_weights)
            # grads = jacrev(self.losses)(params, batch, step, *args)

            # Compute the grad norm of each loss
            grad_norm_dict = {}
            for key, value in grads.items():
                flattened_grad = flatten_pytree(value)
                grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)

            # Compute the mean of grad norms over all losses
            mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))
            # Grad Norm Weighting
            w = tree_map(
                lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict
            )

        elif self.config.weighting.scheme == "ntk":
            # Compute the diagonal of the NTK of each loss
            ntk = self.compute_diag_ntk(params, batch, *args)

            # Compute the mean of the diagonal NTK corresponding to each loss
            mean_ntk_dict = tree_map(lambda x: jnp.mean(x), ntk)

            # Compute the average over all ntk means
            mean_ntk = jnp.mean(jnp.stack(tree_leaves(mean_ntk_dict)))
            # NTK Weighting
            w = tree_map(lambda x: (mean_ntk / (x + 1e-5 * mean_ntk)), mean_ntk_dict)

        return w

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, state.step, *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        # grads = grad(self.loss)(state.params, state.weights, batch, state.step, *args)
        # grads = lax.pmean(grads, "batch")

        # Extract RBA weights
        rba_weights = state.rba_weights
        
        # Compute loss, residuals, and gradients in one forward pass
        (total_loss, residuals), grads = value_and_grad(self.loss, has_aux=True)(
            state.params, 
            state.weights, 
            batch, 
            state.step, 
            rba_weights
        )
        
        # Clip gradients
        grads = tree_map(
            lambda g: jnp.clip(g, -self.config.training.grad_clip, self.config.training.grad_clip),
            grads
        )

        # Update RBA weights
        # if self.config.weighting.use_rba == True and state.step > self.config.weighting.rba_warmup:
        #     rba_weights = self.update_rba_weights(residuals, rba_weights)
        do_update = self.config.weighting.use_rba and (state.step > self.config.weighting.rba_warmup)
        rba_weights = lax.cond(
            do_update,
            lambda _: self.update_rba_weights(residuals, rba_weights),
            lambda _: rba_weights,
            operand=None
        )

        state = state.apply_rba_weights(rba_weights)
        
        state = state.apply_gradients(grads=grads)
        return state


class ForwardIVP(PINN):
    def __init__(self, config):
        super().__init__(config)

        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T


class ForwardBVP(PINN):
    def __init__(self, config):
        super().__init__(config)
