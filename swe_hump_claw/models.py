from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap
from jax.tree_util import tree_map
from flax import jax_utils
import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator

import logging

from utils import g_schedule_step, g_schedule_sigmoid


class SWE2D_NC(ForwardIVP):
    def __init__(self, config, u0, h0, t_star, x_star, bc_coords, Froude_star, g, g_values, manning):
        super().__init__(config)
        logger = logging.getLogger()

        self.u0 = u0
        self.h0 = h0
        self.t_star = t_star
        self.x_star = x_star
        self.bc_coords = bc_coords
        self.Froude_star = Froude_star
        self.g = g
        self.g_values = g_values
        self.manning = manning  # Manning roughness coefficient
        self.visc = self.config.nondim.visc  # Numerical viscosity coefficient

        self.u0_pred_fn = vmap(self.u_net, (None, None, 0))
        self.h0_pred_fn = vmap(self.h_net, (None, None, 0))
        self.u_bc_pred_fn = vmap(self.u_net, (None, 0, 0))
        self.h_bc_pred_fn = vmap(self.h_net, (None, 0, 0))
        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.h_pred_fn = vmap(vmap(self.h_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, None))

        # Initialize RBA weights
        initial_rba_weights = jnp.ones(config.training.batch_size_per_device) # Start with uniform weights
        replicated_rba_weights = jax_utils.replicate(initial_rba_weights)
        self.state = self.state.replace(rba_weights=replicated_rba_weights)

        
    def neural_net(self, params, t, x):
        inputs = jnp.stack([t, x])
        _, outputs = self.state.apply_fn(params, inputs)

        # Start with an initial state of the channel flow
        u = outputs[0]
        h = outputs[1]

        return u, h
    

    def u_net(self, params, t, x):
        u, _ = self.neural_net(params, t, x)
        return u
    

    def h_net(self, params, t, x):
        _, h = self.neural_net(params, t, x)
        return h


    def bathymetry(self, x):
        """
        Returns a Gaussian-shaped bathymetry matching bathymetry_data_gen:
        0.8 * exp(-x²/(0.2)²) - 1.0
        Maintains non-dimensional consistency using L_star and H_star
        """
        # Physical parameters (dimensional values)
        amp_physical = 0.8    # Amplitude in meters
        width_physical = 0.2  # Width in meters
        shift_physical = 1.0  # Vertical shift in meters

        if self.config.nondim.nondimensionalize:
            # Convert to non-dimensional parameters
            L_star = self.config.nondim.L_star
            H_star = self.config.nondim.H_star
            
            amp = amp_physical / H_star
            width = width_physical / L_star
            shift = shift_physical / H_star
        else:
            amp = amp_physical
            width = width_physical
            shift = shift_physical

        # Compute Gaussian bathymetry
        bath = amp * jnp.exp(-x**2 / width**2) - shift
        return bath


    def r_net(self, params, t, x, step):
        u, h = self.neural_net(params, t, x)

        u_t = grad(self.u_net, argnums=1)(params, t, x)
        h_t = grad(self.h_net, argnums=1)(params, t, x)
        
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        h_x = grad(self.h_net, argnums=2)(params, t, x)

        if self.config.setup.use_visc == True:
            u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)

            # Clip extreme values
            u_xx = lax.clamp(-self.config.training.grad_clip, u_xx, self.config.training.grad_clip) 
            # jax.debug.print("u_xx: max={max}, min={min}, mean={mean}", max=jnp.max(u_xx), min=jnp.min(u_xx), mean=jnp.mean(u_xx))

            visc_x = self.visc * (u_xx)
        else:
            visc_x = 0
            # visc_y = 0

        if self.config.training.grad_clip != None:
            u_t = lax.clamp(-self.config.training.grad_clip, u_t, self.config.training.grad_clip) 
            h_t = lax.clamp(-self.config.training.grad_clip, h_t, self.config.training.grad_clip) 
            u_x = lax.clamp(-self.config.training.grad_clip, u_x, self.config.training.grad_clip) 
            h_x = lax.clamp(-self.config.training.grad_clip, h_x, self.config.training.grad_clip) 

        # Bathymetry gradients
        b = self.bathymetry(x)
        b_x = grad(self.bathymetry, argnums=0)(x)
        # jax.debug.print("x: {x}",x=x)
        # jax.debug.print("b: {b}",b=b)
        # jax.debug.print("b_x: {b_x}",b_x=b_x)

        # Gradually increase g with a scheduler
        if self.config.training.g_schedule == "step":
            g = self.g_values[step]
        elif self.config.training.g_schedule == "sigmoid":
            g = self.g_values[step]
        else:
            g = self.g
        # jax.debug.print("g: {g}",g=g)

        froude_star = self.config.nondim.U_star / jnp.sqrt(g * self.config.nondim.H_star)

        # PDE residual
        # Continuity equation (mass conservation)
        rc = h_t + (u * h_x) + (h * (u_x))
        
        # Momentum equation in the x-direction
        ru = u_t + (u * u_x) + (1 / froude_star**2) * (h_x) + (1 / froude_star**2) * (b_x) - visc_x

        # jax.debug.print("rc: {rc}",rc=rc)
        # jax.debug.print("ru: {ru}",ru=ru)
        return ru, rc
    
    def ru_net(self, params, t, x):
        ru, _ = self.r_net(params, t, x)
        return ru

    def rc_net(self, params, t, x):
        _, rc = self.r_net(params, t, x)
        return rc

    # def u_out_net(self, params, t, x, y):
    #     _, _, _, u_out, _ = self.r_net(params, t, x, y)
    #     return u_out

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch, step, rba_weights):
        # Stop gradients on the batch during L-BFGS phase
        if self.config.training.use_lbfgs == True and self.config.training.current_optim == "lbfgs":
            batch = lax.stop_gradient(batch)  # Fix collocation points for L-BFGS

        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rc_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], step
        )

        # Apply RBA weights
        if self.config.weighting.use_rba:
            ru_pred = ru_pred * rba_weights
            rc_pred = rc_pred * rba_weights
            # jax.debug.print("ru_pred res_and_w: max={max}, min={min}, mean={mean}", max=jnp.max(ru_pred), min=jnp.min(ru_pred), mean=jnp.mean(ru_pred))

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rc_gamma])
        # jax.debug.print("gamma ru min: {ru_gamma_min} | ru mean: {ru_gamma_mean} || rc min: {rc_gamma_min} | rc mean: {rc_gamma_mean}",ru_gamma_min=ru_gamma.min(0),ru_gamma_mean=ru_gamma.mean(0),rc_gamma_min=rc_gamma.min(0),rc_gamma_mean=rc_gamma.mean(0))
        gamma = gamma.min(0)

        return ru_l, rc_l, gamma
    

    @partial(jit, static_argnums=(0,))
    def update_rba_weights(self, residuals, current_rba_weights):
        """Update rule: λ^{k+1} = γ*λ^k + η*normalized_residuals"""
        ru_pred_raw, rc_pred_raw = residuals
        combined_res = ru_pred_raw + rc_pred_raw
        
        max_res = jnp.max(jnp.abs(combined_res)) #+ 1e-8
        normalized_res = jnp.abs(combined_res) / max_res

        rba_weights = self.config.weighting.rba_gamma * current_rba_weights + self.config.weighting.rba_eta * normalized_res

        return rba_weights
    

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, step, rba_weights):
        # Stop gradients on the batch during L-BFGS phase
        if self.config.training.use_lbfgs == True and self.config.training.current_optim == "lbfgs":
            batch = lax.stop_gradient(batch)  # Fix collocation points for L-BFGS

        # IC loss
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star) # (21,)
        h0_pred = self.h0_pred_fn(params, 0.0, self.x_star) # (21,)
        u0_loss = jnp.mean((u0_pred - self.u0) ** 2)
        h0_loss = jnp.mean((h0_pred - self.h0) ** 2)

        # Solid wall BC loss: u=0 on left and right walls
        if 'bc' in self.config.weighting.init_weights:
            # Left wall (x=x0)
            u_left_pred = self.u_bc_pred_fn(
                params, 
                self.bc_coords["left_wall"][:, 0],  # time
                self.bc_coords["left_wall"][:, 1],   # x (fixed at x0)
            )
            loss_left = jnp.mean(u_left_pred**2)  # Enforce u=0

            # Right wall (x=x1)
            u_right_pred = self.u_bc_pred_fn(
                params, 
                self.bc_coords["right_wall"][:, 0],  # time
                self.bc_coords["right_wall"][:, 1],   # x (fixed at x1)
            )
            loss_right = jnp.mean(u_right_pred**2)  # Enforce u=0

            bc_loss = loss_left + loss_right

        # Apply RBA weights to residuals
        ru_pred_raw, rc_pred_raw = self.r_pred_fn(params, batch[:, 0], batch[:, 1], step)
        # if self.config.weighting.use_rba:
        #     ru_pred = ru_pred_raw * rba_weights
        #     rc_pred = rc_pred_raw * rba_weights
        #     jax.debug.print("ru_pred losses: max={max}, min={min}, mean={mean}", max=jnp.max(ru_pred), min=jnp.min(ru_pred), mean=jnp.mean(ru_pred))
        # residuals = ru_pred + rc_pred

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rc_l, gamma = self.res_and_w(params, batch, step, rba_weights)
            ru_loss = jnp.mean(gamma * ru_l)
            rc_loss = jnp.mean(gamma * rc_l)

        else:
            ru_pred, rc_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], step
            )

            if self.config.weighting.use_rba:
                ru_pred = ru_pred * rba_weights
                rc_pred = rc_pred * rba_weights

            ru_loss = jnp.mean(ru_pred**2)
            rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_ic": u0_loss,
            "h_ic": h0_loss,
            "ru": ru_loss,
            "rc": rc_loss,
        }
        if 'bc' in self.config.weighting.init_weights:
            loss_dict['bc'] = bc_loss

        # if 'in_u_bc' in self.config.weighting.init_weights:
        #     loss_dict['in_u_bc'] = u_in_loss

        # if 'outflow_bc' in self.config.weighting.init_weights:
        #     loss_dict['outflow_bc'] = outflow_bc_loss

        return loss_dict, (ru_pred_raw, rc_pred_raw)
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, u_ref, h_ref):
        u_pred = self.u_pred_fn(params, t, x)
        h_pred = self.h_pred_fn(params, t, x)

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        h_error = jnp.linalg.norm(h_pred - h_ref) / jnp.linalg.norm(h_ref)

        return u_error, h_error
     

class SWEEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref, h_ref):
        # Reduce evaluation size
        ratio = self.config.training.ratio
        x_star = self.model.x_star[::ratio]
        u_ref = u_ref[:,::ratio]
        h_ref = h_ref[:,::ratio]
        
        # Compute the L2 errors for h
        u_error, h_error = self.model.compute_l2_error(params, self.model.t_star, x_star, u_ref, h_ref)
        self.log_dict["l2_u_error"] = u_error
        self.log_dict["l2_h_error"] = h_error

    def __call__(self, state, batch, u_ref, h_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, causal_weight = self.model.res_and_w(state.params, batch, state.step, state.rba_weights)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, h_ref)

        # Log nonlinearities for Pirate
        if self.config.logging.log_nonlinearities:
            layer_keys = [
                key
                for key in state.params["params"].keys()
                if key.endswith(
                    tuple(
                        [f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]
                    )
                )
            ]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params["params"][key]["alpha"]

        # Compute total loss
        total_loss, _ = self.model.loss(state.params, state.weights, batch, state.step, state.rba_weights)
        self.log_dict["total_loss"] = total_loss

        return self.log_dict