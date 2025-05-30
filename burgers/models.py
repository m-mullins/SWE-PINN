from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, debug

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
from utils import get_tree_size_mb

class Burgers(ForwardIVP):
    def __init__(self, config, u0, t_star, x_star):
        super().__init__(config)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        # Predictions over a grid
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0, None)), (None, 0, None, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def grad_net(self, params, t, x):
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_t, u_x, u_xx

    def r_net(self, params, t, x, step):
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        # return u_t + u * u_x - 0.01 / jnp.pi * u_xx
        return u_t + u * u_x - self.config.nu * u_xx

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch, step, rba_weights):
        "Compute residuals and weights for causal training"
        # Sort time coordinates
        t_sorted = batch[:, 0].sort()
        r_pred = vmap(self.r_net, (None, 0, 0, None))(params, t_sorted, batch[:, 1], step)

        if self.config.weighting.use_rba:
            r_pred = r_pred * rba_weights

        # Split residuals into chunks
        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred**2, axis=1)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w
    
    @partial(jit, static_argnums=(0,))
    def update_rba_weights(self, residuals, current_rba_weights):
        """Update rule: λ^{k+1} = γ*λ^k + η*normalized_residuals"""
        ru_pred_raw = residuals
        combined_res = ru_pred_raw
        
        max_res = jnp.max(jnp.abs(combined_res)) #+ 1e-8
        normalized_res = jnp.abs(combined_res) / max_res

        rba_weights = self.config.weighting.rba_gamma * current_rba_weights + self.config.weighting.rba_eta * normalized_res

        return rba_weights

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, step, rba_weights):
        # Initial condition loss
        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.u0 - u_pred) ** 2)

        # Apply RBA weights to residuals
        r_pred_raw = self.r_pred_fn(params, batch[:, 0], batch[:, 1], step)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch, step, rba_weights)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])

            if self.config.weighting.use_rba:
                r_pred = r_pred * rba_weights

            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict, (r_pred_raw)

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        ics_ntk = vmap(ntk_fn, (None, None, None, 0))(
            self.u_net, params, self.t0, self.x_star
        )

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            # sort the time step for causal loss
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1]]).T
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )
            res_ntk = res_ntk.reshape(self.num_chunks, -1)  # shape: (num_chunks, -1)
            res_ntk = jnp.mean(
                res_ntk, axis=1
            )  # average convergence rate over each chunk
            _, casual_weights = self.res_and_w(params, batch)
            res_ntk = res_ntk * casual_weights  # multiply by causal weights
        else:
            res_ntk = vmap(ntk_fn, (None, None, 0, 0))(
                self.r_net, params, batch[:, 0], batch[:, 1]
            )

        ntk_dict = {"ics": ics_ntk, "res": res_ntk}

        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error



class BurgersEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star, self.model.x_star)
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(u_pred.T, cmap="jet")
        self.log_dict["u_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch, state.step, state.rba_weights)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)

        return self.log_dict
