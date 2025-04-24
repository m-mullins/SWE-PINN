from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, hessian
from jax.tree_util import tree_map
from flax import jax_utils
import optax

from jaxpi import archs
from jaxpi.models import ForwardBVP, ForwardIVP
from jaxpi.utils import ntk_fn
from jaxpi.evaluator import BaseEvaluator

import logging

from utils import g_schedule_step, g_schedule_sigmoid, clipped_second_derivative


class SWE2D_NC(ForwardIVP):
    def __init__(self, config, u0, v0, h0, t_star, x_star, y_star, bc_coords, Froude_star, g, g_values, manning):
        super().__init__(config)
        logger = logging.getLogger()

        self.u0 = u0
        self.v0 = v0
        self.h0 = h0
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.bc_coords = bc_coords
        self.Froude_star = Froude_star
        self.g = g
        self.g_values = g_values
        self.manning = manning  # Manning roughness coefficient
        self.visc = self.config.nondim.visc  # Numerical viscosity coefficient
        

        self.u0_pred_fn = vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None))
        self.v0_pred_fn = vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None))
        self.h0_pred_fn = vmap(vmap(self.h_net, (None, None, None, 0)), (None, None, 0, None))
        self.u_bc_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_bc_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.h_bc_pred_fn = vmap(self.h_net, (None, 0, 0, 0))
        self.u_pred_fn = vmap(vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.h_pred_fn = vmap(vmap(vmap(self.h_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, None))

        # Initialize RBA weights
        initial_rba_weights = jnp.ones(config.training.batch_size_per_device) # Start with uniform weights
        replicated_rba_weights = jax_utils.replicate(initial_rba_weights)
        self.state = self.state.replace(rba_weights=replicated_rba_weights)

    def neural_net(self, params, t, x, y):
        inputs = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, inputs)

        # Start with an initial state of the channel flow
        u = outputs[0]
        v = outputs[1]
        h = outputs[2]

        # logger = logging.getLogger()
        # logger.info("Neural net outputs: ")
        # jax.debug.print("t: {t}",t=t)
        # jax.debug.print("x: {x}",x=x)
        # jax.debug.print("y: {y}",y=y)
        # jax.debug.print("u: {u}",u=u)
        # jax.debug.print("v: {v}",v=v)
        # jax.debug.print("h: {h}",h=h)
        # u_value = jax.device_get(u)  # Get actual value if needed
        # logger.info(f"u_value: {u_value}")

        return u, v, h

    def u_net(self, params, t, x, y):
        u, _, _ = self.neural_net(params, t, x, y)
        return u

    def v_net(self, params, t, x, y):
        _, v, _ = self.neural_net(params, t, x, y)
        return v

    def h_net(self, params, t, x, y):
        _, _, h = self.neural_net(params, t, x, y)
        return h
    
    def bathymetry(self,x,y):
        """Returns the bathymetry elevation at point (x,y)."""
        if self.config.nondim.nondimensionalize == True:
            x = x * self.config.nondim.L_star
            y = y * self.config.nondim.L_star
        r2 = (x-1.)**2 + (y-0.5)**2
        return jnp.zeros_like(x)

    def r_net(self, params, t, x, y, step):
        u, v, h = self.neural_net(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        v_t = grad(self.v_net, argnums=1)(params, t, x, y)
        h_t = grad(self.h_net, argnums=1)(params, t, x, y)
        
        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        h_x = grad(self.h_net, argnums=2)(params, t, x, y)

        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)
        h_y = grad(self.h_net, argnums=3)(params, t, x, y)

        if self.config.setup.use_visc == True:
            # u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
            # u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)

            # v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y)
            # v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)

            safe_u_xx_fn = clipped_second_derivative(self.u_net, argnum=2, clip_val=self.config.training.grad_clip)
            safe_u_yy_fn = clipped_second_derivative(self.u_net, 3, self.config.training.grad_clip)
            safe_v_xx_fn = clipped_second_derivative(self.v_net, 2, self.config.training.grad_clip)
            safe_v_yy_fn = clipped_second_derivative(self.v_net, 3, self.config.training.grad_clip)
            u_xx = safe_u_xx_fn(params, t, x, y)
            u_yy = safe_u_yy_fn(params, t, x, y)
            v_xx = safe_v_xx_fn(params, t, x, y)
            v_yy = safe_v_yy_fn(params, t, x, y)

            # Clip extreme values
            u_xx = lax.clamp(-self.config.training.grad_clip, u_xx, self.config.training.grad_clip) 
            u_yy = lax.clamp(-self.config.training.grad_clip, u_yy, self.config.training.grad_clip) 
            v_xx = lax.clamp(-self.config.training.grad_clip, v_xx, self.config.training.grad_clip) 
            v_yy = lax.clamp(-self.config.training.grad_clip, v_yy, self.config.training.grad_clip) 
            # jax.debug.print("u_xx: max={max}, min={min}, mean={mean}", max=jnp.max(u_xx), min=jnp.min(u_xx), mean=jnp.mean(u_xx))
            # jax.debug.print("u_yy: max={max}, min={min}, mean={mean}", max=jnp.max(u_yy), min=jnp.min(u_yy), mean=jnp.mean(u_yy))

            visc_x = self.visc * (u_xx + u_yy)
            visc_y = self.visc * (v_xx + v_yy)
        else:
            visc_x = 0
            visc_y = 0

        if self.config.training.grad_clip != None:
            u_t = lax.clamp(-self.config.training.grad_clip, u_t, self.config.training.grad_clip)
            v_t = lax.clamp(-self.config.training.grad_clip, v_t, self.config.training.grad_clip)
            h_t = lax.clamp(-self.config.training.grad_clip, h_t, self.config.training.grad_clip)
            u_x = lax.clamp(-self.config.training.grad_clip, u_x, self.config.training.grad_clip)
            v_x = lax.clamp(-self.config.training.grad_clip, v_x, self.config.training.grad_clip)
            h_x = lax.clamp(-self.config.training.grad_clip, h_x, self.config.training.grad_clip)
            u_y = lax.clamp(-self.config.training.grad_clip, u_y, self.config.training.grad_clip) 
            v_y = lax.clamp(-self.config.training.grad_clip, v_y, self.config.training.grad_clip) 
            h_y = lax.clamp(-self.config.training.grad_clip, h_y, self.config.training.grad_clip) 

        # Bathymetry gradients
        # b = self.bathymetry(x,y)
        # b_x = grad(self.bathymetry, argnums=0)(x, y)
        # b_y = grad(self.bathymetry, argnums=1)(x, y)
        # jax.debug.print("t: {t}",t=t)
        # jax.debug.print("x: {x}",x=x)
        # jax.debug.print("y: {y}",y=y)
        # jax.debug.print("b: {b}",b=b)
        # jax.debug.print("b_x: {b_x}",b_x=b_x)

        # # Friction terms
        # sf_x = 0 # (jnp.square(self.config.setup.manning) * u * jnp.sqrt(jnp.square(u) + jnp.square(v))) / (jnp.abs(h) ** (4/3))
        # sf_y = 0 # (jnp.square(self.config.setup.manning) * v * jnp.sqrt(jnp.square(u) + jnp.square(v))) / (jnp.abs(h) ** (4/3))
        # # jax.debug.print("sf_x: {sf_x}",sf_x=sf_x)
        # # jax.debug.print("h**: {hpp}",hpp=(h ** (4/3)))
        # # jax.debug.print("manning: {manning}",manning=self.manning)

        # # Viscosity terms
        # svis_x = 0 # self.config.setup.nu * (u_xx + u_yy)
        # svis_y = 0 # self.config.setup.nu * (v_xx + v_yy)

        # # Source terms of the equations
        # s0 = 0
        # s1 = 0 #-self.config.setup.g * (b_x + sf_x)
        # s2 = 0 #-self.config.setup.g * (b_y + sf_y)

        # Gradually increase g with a scheduler
        # jax.debug.print("step: {step}",step=step)
        if self.config.training.g_schedule == "step":
            g = self.g_values[step]
        elif self.config.training.g_schedule == "sigmoid":
            g = self.g_values[step]
        else:
            g = self.g
        # jax.debug.print("g: {g}",g=g)

        froude_star = self.config.nondim.U_star / jnp.sqrt(g * self.config.nondim.H_star)

        # Friction terms
        # lambda_star = - g * self.manning**2 * self.config.nondim.L_star / (self.config.nondim.H_star)**(4/3)
        # h_pos = jnp.where(h <= 0, 1e-3, h) # Negative h returns NaN in sf_x, sf_y calc
        # sf_x = lambda_star * u * jnp.sqrt(u**2 + v**2) / (h_pos)**(4/3)
        # sf_y = lambda_star * v * jnp.sqrt(u**2 + v**2) / (h_pos)**(4/3)
        # jax.debug.print("lambda_star: {lambda_star}",lambda_star=lambda_star)
        # jax.debug.print("sf_y: {sf_y}",sf_y=sf_y)
        # jax.debug.print("u: {u}",u=u)
        # jax.debug.print("v: {v}",v=v)
        # jax.debug.print("h: {h}",h=h)
        # jax.debug.print("h_pos: {h_pos}",h_pos=h_pos)
        # jax.debug.print("sf_x: {sf_x}",sf_x=sf_x)

        # PDE residual
        # Continuity equation (mass conservation)
        rc = h_t + (u * h_x) + (v * h_y) + (h * (u_x + v_y))
        
        # Momentum equation in the x-direction
        # ru = u_t + (u * u_x) + (v * u_y) + (self.config.setup.g * h_x) - s1 - svis_x
        ru = u_t + (u * u_x) + (v * u_y) + (1 / froude_star**2) * (h_x) - visc_x #- sf_x

        # Momentum equation in the y-direction
        # rv = v_t + (u * v_x) + (v * v_y) + (self.config.setup.g * h_y) - s2 - svis_y
        rv = v_t + (u * v_x) + (v * v_y) + (1 / froude_star**2) * (h_y) - visc_y #- sf_y
        # jax.debug.print("rc: {rc}",rc=rc)
        # jax.debug.print("ru: {ru}",ru=ru)
        # jax.debug.print("rv: {rv}",rv=rv)
        return ru, rv, rc
    
    def ru_net(self, params, t, x, y):
        ru, _, _ = self.r_net(params, t, x, y)
        return ru

    def rv_net(self, params, t, x, y):
        _, rv, _ = self.r_net(params, t, x, y)
        return rv

    def rc_net(self, params, t, x, y):
        _, _, rc = self.r_net(params, t, x, y)
        return rc

    # def u_out_net(self, params, t, x, y):
    #     _, _, _, u_out, _ = self.r_net(params, t, x, y)
    #     return u_out

    # def v_out_net(self, params, t, x, y):
    #     _, _, _, _, v_out = self.r_net(params, t, x, y)
    #     return v_out

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch, step, rba_weights):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        ru_pred, rv_pred, rc_pred = self.r_pred_fn(
            params, t_sorted, batch[:, 1], batch[:, 2], step
        )

        # Apply RBA weights
        if self.config.weighting.use_rba:
            ru_pred = ru_pred * rba_weights
            rv_pred = rv_pred * rba_weights
            rc_pred = rc_pred * rba_weights

        ru_pred = ru_pred.reshape(self.num_chunks, -1)
        rv_pred = rv_pred.reshape(self.num_chunks, -1)
        rc_pred = rc_pred.reshape(self.num_chunks, -1)

        ru_l = jnp.mean(ru_pred**2, axis=1)
        rv_l = jnp.mean(rv_pred**2, axis=1)
        rc_l = jnp.mean(rc_pred**2, axis=1)

        ru_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ ru_l)))
        rv_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rv_l)))
        rc_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rc_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma])
        gamma = gamma.min(0)

        return ru_l, rv_l, rc_l, gamma
    
    @partial(jit, static_argnums=(0,))
    def update_rba_weights(self, residuals, current_rba_weights):
        """Update rule: λ^{k+1} = γ*λ^k + η*normalized_residuals"""
        ru_pred_raw, rv_pred_raw, rc_pred_raw = residuals
        combined_res = ru_pred_raw + rv_pred_raw + rc_pred_raw
        
        max_res = jnp.max(jnp.abs(combined_res)) #+ 1e-8
        normalized_res = jnp.abs(combined_res) / max_res

        rba_weights = self.config.weighting.rba_gamma * current_rba_weights + self.config.weighting.rba_eta * normalized_res

        return rba_weights

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, step, rba_weights):
        # IC loss
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star, self.y_star) # (21,21)
        v0_pred = self.v0_pred_fn(params, 0.0, self.x_star, self.y_star) # (21,21)
        h0_pred = self.h0_pred_fn(params, 0.0, self.x_star, self.y_star) # (21,21)
        u0_loss = jnp.mean((u0_pred - self.u0) ** 2)
        v0_loss = jnp.mean((v0_pred - self.v0) ** 2)
        h0_loss = jnp.mean((h0_pred - self.h0) ** 2)

        # BC loss
        if 'slip_bc' in self.config.weighting.init_weights:
            # Slip BC: u=0 on right wall, v=0 on top wall
            # Right wall (x=x1)
            u_right_pred = self.u_bc_pred_fn(
                params, 
                self.bc_coords["right"][:, 0],  # time
                self.bc_coords["right"][:, 1],  # x
                self.bc_coords["right"][:, 2],  # y
            )
            loss_right = jnp.mean(u_right_pred**2)  # Enforce u=0

            # Top wall (y=y1)
            v_top_pred = self.v_bc_pred_fn(
                params, 
                self.bc_coords["top"][:, 0],  # time
                self.bc_coords["top"][:, 1],  # x
                self.bc_coords["top"][:, 2],  # y
            )
            loss_top = jnp.mean(v_top_pred**2)  # Enforce v=0

            slip_bc_loss = loss_right + loss_top

            # u_bc_pred = self.u_bc_pred_fn(params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])
            # v_bc_pred = self.v_bc_pred_fn(params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])
            # h_bc_pred = self.h_bc_pred_fn(params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])
            # u_bc_loss = jnp.mean((u_bc_pred - self.bc_values[:, 0]) ** 2)
            # v_bc_loss = jnp.mean((v_bc_pred - self.bc_values[:, 1]) ** 2)
            # h_bc_loss = jnp.mean((h_bc_pred - self.bc_values[:, 2]) ** 2)
        
        # Apply RBA weights to residuals
        ru_pred_raw, rv_pred_raw, rc_pred_raw = self.r_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2], step)

        # residual loss
        if self.config.weighting.use_causal == True:
            ru_l, rv_l, rc_l, gamma = self.res_and_w(params, batch, step, rba_weights)
            ru_loss = jnp.mean(gamma * ru_l)
            rv_loss = jnp.mean(gamma * rv_l)
            rc_loss = jnp.mean(gamma * rc_l)

        else:
            ru_pred, rv_pred, rc_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2], step
            )

            if self.config.weighting.use_rba:
                ru_pred = ru_pred * rba_weights
                rv_pred = rv_pred * rba_weights
                rc_pred = rc_pred * rba_weights

            ru_loss = jnp.mean(ru_pred**2)
            rv_loss = jnp.mean(rv_pred**2)
            rc_loss = jnp.mean(rc_pred**2)

        loss_dict = {
            "u_ic": u0_loss,
            "v_ic": v0_loss,
            "h_ic": h0_loss,
            "ru": ru_loss,
            "rv": rv_loss,
            "rc": rc_loss,
        }
        if 'slip_bc' in self.config.weighting.init_weights:
            loss_dict['slip_bc'] = slip_bc_loss
            # loss_dict['v_bc'] = v_bc_loss
            # loss_dict['h_bc'] = h_bc_loss
        # if 'outflow_bc' in self.config.weighting.init_weights:
        #     loss_dict['outflow_bc'] = outflow_bc_loss

        return loss_dict, (ru_pred_raw, rv_pred_raw, rc_pred_raw)
    
    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, u_ref, v_ref, h_ref):
        u_pred = self.u_pred_fn(params, t, x, y)
        v_pred = self.v_pred_fn(params, t, x, y)
        h_pred = self.h_pred_fn(params, t, x, y)

        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)
        h_error = jnp.linalg.norm(h_pred - h_ref) / jnp.linalg.norm(h_ref)

        return u_error, v_error, h_error
     

class SWEEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    # def log_preds(self, params, x_star, y_star):
    #     u_pred = vmap(vmap(model.u_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     v_pred = vmap(vmap(model.v_net, (None, None, 0)), (None, 0, None))(params, x_star, y_star)
    #     U_pred = jnp.sqrt(u_pred ** 2 + v_pred ** 2)
    #
    #     fig = plt.figure()
    #     plt.pcolor(U_pred.T, cmap='jet')
    #     log_dict['U_pred'] = fig
    #     fig.close()

    def log_errors(self, params, u_ref, v_ref, h_ref):
        # Reduce evaluation size
        ratio = self.config.training.ratio
        x_star = self.model.x_star[::ratio]
        y_star = self.model.y_star[::ratio]
        u_ref = u_ref[:,::ratio,::ratio]
        v_ref = v_ref[:,::ratio,::ratio]
        h_ref = h_ref[:,::ratio,::ratio]

        # Compute the L2 errors for h
        u_error, v_error, h_error = self.model.compute_l2_error(params, self.model.t_star, x_star, y_star, u_ref, v_ref, h_ref)
        self.log_dict["l2_u_error"] = u_error
        self.log_dict["l2_v_error"] = v_error
        self.log_dict["l2_h_error"] = h_error

    def __call__(self, state, batch, u_ref, v_ref, h_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, _, _, causal_weight = self.model.res_and_w(state.params, batch, state.step, state.rba_weights)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref, v_ref, h_ref)

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