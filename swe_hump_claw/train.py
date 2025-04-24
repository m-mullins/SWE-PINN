import functools
from functools import partial
import time
import os
from types import SimpleNamespace

from absl import logging

import jax
import optax
import optimistix as optx

import jax.numpy as jnp
from jax import random, vmap, pmap, local_device_count, lax
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from flax import jax_utils
from jaxopt import LBFGS, BFGS, BacktrackingLineSearch

import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import ml_collections

import wandb
import platform

import models

from jaxpi.samplers import UniformSampler, BaseSampler, SpaceSampler, TimeSpaceSampler, FixedSampler, FixedRandomSampler, StructuredRandomSampler, RADSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

from utils import get_dataset, convert_config_to_dict, get_bc_coords, g_schedule_step, g_schedule_sigmoid, BFGSTrustRegion, count_params, plot_rba_weights, plot_colloc_pts, plot_residuals


def train_one_window(config, workdir, model, samplers, idx, u_ref, h_ref):
    # Initialize evaluator
    evaluator = models.SWEEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    step_offset = idx * config.training.max_steps

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    # Phase 1: ADAM optimizer
    for step in range(config.training.max_steps):
        if config.weighting.use_rba == True:
            if config.weighting.rba_sampler == "fixed":
                batch = next(samplers['fixed_sampler'])
            else:
                batch = next(samplers['structured_random_sampler'])
        elif config.weighting.use_rad == True:
            if step % config.weighting.rad_update_every_steps == 0:
                batch, residuals, s0 = samplers['rad_sampler'](model.state)
        else:
            batch = next(samplers['res_sampler'])   # shape (num_devices, bs, num_dim)
        model.state = model.step(model.state, batch)

        # jax.debug.print(
        #     "Train    RBA max: {max} | RBA min: {min} | RBA mean: {mean}",
        #     max=jnp.max(model.state.rba_weights),
        #     min=jnp.min(model.state.rba_weights),
        #     mean=jnp.mean(model.state.rba_weights)
        # )

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_log = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch_log, u_ref, h_ref)

                # # Compute total loss
                # total_loss = model.loss(state.params, state.weights, batch, state.step)
                # log_dict["total_loss"] = total_loss

                # Get global step if using curriculum training
                if config.transfer.curriculum == True:
                    step_logged = step + config.logging.global_step
                    
                    # Skip last step except for last dataset in curriculum
                    if (step+1) != config.training.max_steps:
                        wandb.log(log_dict, step_logged + step_offset)
                        end_time = time.time()
                        logger.log_iter(step_logged, start_time, end_time, log_dict)

                # Regular
                else:
                    wandb.log(log_dict, step + step_offset)

                    end_time = time.time()
                    # Report training metrics
                    logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Save checkpoint
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

        # Log RBA weights
        if step in config.logging.log_rba_every_steps and config.weighting.use_rba:
            rba_weights = jax.device_get(tree_map(lambda x: x[0], model.state.rba_weights))
            
            plot_rba_weights(config, workdir, rba_weights, batch, step + step_offset)

        # Log collocation points
        if step in config.logging.log_colloc_every_steps and config.weighting.use_rad:   
            dom = jnp.array([[model.t_star[0], model.t_star[-1]], [model.x_star[0], model.x_star[-1]]])
            plot_colloc_pts(config, workdir, batch, step + step_offset, dom)

        # Log residuals
        if step in config.logging.log_colloc_every_steps and config.weighting.use_rad:   
            dom = jnp.array([[model.t_star[0], model.t_star[-1]], [model.x_star[0], model.x_star[-1]]])
            plot_residuals(config, workdir, residuals[0], s0[0], step + step_offset, dom)

    # Phase 2: L-BFGS with fixed collocation points
    if config.training.use_lbfgs:
        logger.info(f"Starting L-BFGS train phase")
        config.training.current_optim = "lbfgs"

        # Convert JAXPI's replicated state to single-device
        unreplicated_state = jax_utils.unreplicate(model.state)
        params = unreplicated_state.params
        weights = unreplicated_state.weights
        rba_weights = unreplicated_state.rba_weights

        batch = samplers['fixed_sampler_lbfgs'].fixed_batch

        # Enable double precision, needed for L-BFGS optimization
        jax.config.update("jax_enable_x64", True)
        params = jax.tree_map(lambda x: x.astype(jnp.float64).block_until_ready(), params)
        batch = jax.tree_map(lambda x: x.astype(jnp.float64).block_until_ready(), batch)
        weights = jax.tree_map(lambda x: x.astype(jnp.float64).block_until_ready(), weights)
        rba_weights = jax.tree_map(lambda x: x.astype(jnp.float64).block_until_ready(), rba_weights)
        logger.info(f"Arrays casted to x64")

        # L-BFGS loss function (scalar output)
        def lbfgs_loss_fn(params, weights, batch, rba_weights):
            loss, _ = model.loss(params, weights, batch, step=-1, rba_weights=rba_weights)  # Scalar
            # jax.debug.print("weights: {weights}",weights=weights)
            # jax.debug.print(
            #     "Train lbfgs_loss_fn RBA max: {max} | RBA min: {min} | RBA mean: {mean}",
            #     max=jnp.max(rba_weights),
            #     min=jnp.min(rba_weights),
            #     mean=jnp.mean(rba_weights)
            # )
            return loss
        
        # Strong Wolfe line search loss function (scalar loss with grads output)
        def line_search_loss_fn(params, weights, batch, rba_weights):
            # batch = samplers['fixed_sampler_lbfgs'].fixed_batch
            loss, grads = model.loss_grads(params, weights, batch, step=-1, rba_weights=rba_weights)
            # grad_flat, _ = ravel_pytree(grads)
            # grad_norm = jnp.linalg.norm(grad_flat)
            # jax.debug.print("GRADIENT NORM: {grad_norm}", grad_norm=grad_norm)
            return loss, grads
        
        # Strong Wolfe line-search
        strong_wolfe_linesearch = BacktrackingLineSearch(
            fun=line_search_loss_fn, # Uses (total loss, grads)
            value_and_grad=True,
            condition="strong-wolfe",
            c1=config.training.lbfgs_c1,
            c2=config.training.lbfgs_c2
        )

        # Initialize optimizer and state
        optimizer = LBFGS(fun=lbfgs_loss_fn, maxiter=config.training.lbfgs_max_steps, linesearch=strong_wolfe_linesearch, history_size=config.training.lbfgs_history_size, verbose=False)
        state = optimizer.init_state(params, weights, batch, rba_weights)

        # L-BFGS ptimization loop
        for lbfgs_step in range(config.training.lbfgs_max_steps):
            params, state = optimizer.update(params, state, weights, batch, rba_weights)
            
            # Update RBA weights
            # rba_weights = state.rba_weights
            if config.weighting.use_rba == True:
                _, residuals = model.loss(params, weights, batch, step=-1, rba_weights=rba_weights) # Need to recompute to get residuals in JIT environment
                rba_weights = model.update_rba_weights(residuals, rba_weights)
                # jax.debug.print(
                #     "Train update RBA max: {max} | RBA min: {min} | RBA mean: {mean}",
                #     max=jnp.max(rba_weights),
                #     min=jnp.min(rba_weights),
                #     mean=jnp.mean(rba_weights)
                # )
            # state = state.apply_rba_weights(rba_weights)

            # Update weights if necessary
            if config.weighting.scheme in ["grad_norm", "ntk"]:
                if lbfgs_step % config.weighting.update_every_steps_lbfgs == 0:
                    # Update model state
                    replicated_params = jax_utils.replicate(params)
                    replicated_batch = jax_utils.replicate(batch)
                    model.state = model.state.replace(params=replicated_params)

                    model.state = model.update_weights(model.state, replicated_batch)

                    # Convert JAXPI's replicated state to single-device
                    unreplicated_state = jax_utils.unreplicate(model.state)
                    weights = unreplicated_state.weights

            # Log loss using wandb
            if jax.process_index() == 0:
                if lbfgs_step % config.logging.log_every_steps_lbfgs == 0:
                    # Update model state
                    replicated_params = jax_utils.replicate(params)
                    replicated_batch = jax_utils.replicate(batch)
                    model.state = model.state.replace(params=replicated_params)

                    # Get the first replica of the state and batch
                    state_log = jax.device_get(tree_map(lambda x: x[0], model.state))
                    batch_log = jax.device_get(tree_map(lambda x: x[0], replicated_batch))
                    log_dict = evaluator(state_log, batch_log, u_ref, h_ref)

                    wandb.log(log_dict, step_offset + config.training.max_steps + lbfgs_step)
                    end_time = time.time()

                    # Report training metrics
                    logger.log_iter(lbfgs_step + config.training.max_steps , start_time, end_time, log_dict)
                    start_time = end_time
                    logger.info(f"L-BFGS Step {lbfgs_step}: Total loss = {state.value:.4e}")

        # Final model state update
        optimized_params = params
        replicated_params = jax_utils.replicate(optimized_params)
        model.state = model.state.replace(params=replicated_params)

        # Save checkpoint
        ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize logger
    logger = Logger()

    # Find out if running on pc for dubugging or on HPC without internet access
    if 'microsoft' in platform.uname().release.lower():
        mode = "online"
    else:
        mode = "offline"
    # mode = "offline"
        
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config), tags=wandb_config.tag, notes=wandb_config.notes)
    logger.info(f"wandb initialized {mode}")
    logger.info(f"Number of devices: {jax.device_count()}")

    # Get ref solution
    (h_ref, u_ref, hu_ref, b_ref, t_ref, x_ref, s_ref, g, manning) = get_dataset(config.dataset)
    logger.info(f"g = {g}")
    logger.info(f"Config: {config.wandb.name}")
    logger.info(f"Dataset: {config.dataset}")

    # Nondimensionalization
    if config.nondim.nondimensionalize == True:
        # Nondimensionalization parameters        
        U_star = config.nondim.U_star   # characteristic velocity
        L_star = config.nondim.L_star   # characteristic length
        H_star = config.nondim.H_star   # characteristic height
        T_star = L_star / U_star        # characteristic time
        Froude_star = U_star / jnp.sqrt(g * config.nondim.H_star)
        logger.info(f"Froude* = {Froude_star}")

        # Nondimensionalize the flow field
        t_star = t_ref / T_star  # Non-dimensionalize time
        x_star = x_ref / L_star  # Non-dimensionalize x
        # y_star = y_ref / L_star  # Non-dimensionalize y
        u_star = u_ref / U_star  # Non-dimensionalize velocity in x
        # v_star = v_ref / U_star  # Non-dimensionalize velocity in y
        h_star = h_ref / H_star  # Non-dimensionalize height
        b_star = b_ref / H_star  # Non-dimensionalize bathymetry
    else:
        t_star = t_ref  # Non-dimensionalize time
        x_star = x_ref  # Non-dimensionalize x
        # y_star = y_ref  # Non-dimensionalize y
        u_star = u_ref  # Non-dimensionalize velocity in x
        # v_star = v_ref  # Non-dimensionalize velocity in y
        h_star = h_ref  # Non-dimensionalize height
        b_star = b_ref  # Non-dimensionalize bathymetry
        Froude_star = 1 / jnp.sqrt(g)
        logger.info(f"Froude* = {Froude_star}")

    u0 = u_star[0, :]
    # v0 = v_star[0, :, :]
    h0 = h_star[0, :]

    x0 = x_star[0]
    x1 = x_star[-1]

    # y0 = y_star[0]
    # y1 = y_star[-1]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    dt = round(t[-1] - t[-2],6)
    t1 = t[-1] + dt # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Calculate numerical viscosity
    # if config.nondim.nondimensionalize == True:
    #     config.nondim.visc = U_star**2 * dt / 2
    # else:
    #     config.nondim.visc = dt / 2
    if config.setup.use_visc == True:
        logger.info(f"Numerical viscosity = {config.nondim.visc}")

    # Define bc coords
    bc_coords = get_bc_coords(dom, t)

    # # Inflow boundary conditions
    # inflow_fn = lambda y: parabolic_inflow(y * L_star, config.setup.U_max)

    # Residual function that wraps your model's residual computation
    residual_fn = lambda state, points: model.residuals(state, points)

    keys = random.split(random.PRNGKey(0), config.training.num_time_windows)

    # # Define residual sampler of collocation points
    samplers = {
        "res_sampler": iter(UniformSampler(dom, config.training.batch_size_per_device)),    # ADAM
        "fixed_sampler": iter(FixedSampler(dom, config.training.batch_size_per_device)),    # ADAM
        "structured_random_sampler": iter(StructuredRandomSampler(dom, config.training.batch_size_per_device)), # RBA
        "rad_sampler": RADSampler(dom, config.training.batch_size_per_device, residual_fn, config.weighting.rad_k, config.weighting.rad_c), # RAD
        "fixed_sampler_lbfgs": FixedSampler(dom, config.training.lbfgs_batch_size),         # L-BFGS
        "fixed_random_sampler_lbfgs": FixedRandomSampler(dom, config.training.lbfgs_batch_size),         # L-BFGS
    }

    # Precompute g values for gradual training
    if config.training.g_schedule != None:
        g_values = np.zeros(config.training.max_steps)
        for step in range(config.training.max_steps):

            if config.training.g_schedule == "step":
                g_values[step] = g_schedule_step(step, config.training.g_min, g, config.training.max_steps, n=5)

            elif config.training.g_schedule == "sigmoid":
                g_values[step] = g_schedule_sigmoid(step, config.training.g_min, g, config.training.max_steps, k=10)
        g_values = jnp.array(g_values)
    else:
        g_values = None

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # Get the reference solution for the current time window
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        # v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        h = h_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization
        if config.use_pi_init == True and config.transfer.s2s_pi_init == True:
            logger.info("Use physics-informed initialization...")

            model = models.SWE2D_NC(config, u0, h0, t, x_star, bc_coords, Froude_star, g, g_values, manning)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "initial_condition":
                t_init = t_star[::2]
                downsample = 1
                x = x_star[::downsample]
                # y = y_star[::downsample]

                u_init = u0[::downsample]
                # v_init = v0[::downsample, ::downsample]
                h_init = h0[::downsample]

                t_scaled = t_init / t_init[-1]

                tt, xx = jnp.meshgrid(t_scaled, x, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

                u_init = jnp.tile(u_init, (t_scaled.shape[0], 1))
                # v_init = jnp.tile(v_init, (t_scaled.shape[0], 1, 1))
                h_init = jnp.tile(h_init, (t_scaled.shape[0], 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            u_coeffs, u_res, rank, s = jnp.linalg.lstsq(feat_matrix, u_init.flatten(), rcond=None)
            # v_coeffs, v_res, rank, s = jnp.linalg.lstsq(feat_matrix, v_init.flatten(), rcond=None)
            h_coeffs, h_res, rank, s = jnp.linalg.lstsq(feat_matrix, h_init.flatten(), rcond=None)

            logger.info(f"least square u residuals: {u_res}")
            # logger.info(f"least square v residuals: {v_res}")
            logger.info(f"least square h residuals: {h_res}")

            coeffs = jnp.vstack([u_coeffs, h_coeffs]).T
            config.arch.pi_init = coeffs

            del model, state, params

        # Initialize model
        model = models.SWE2D_NC(config, u0, h0, t, x_star, bc_coords, Froude_star, g, g_values, manning)

        # Count params
        total_params, total_mb = count_params(model.state)
        logger.info(f"Amount of params: {total_params}")
        logger.info(f"Model size: {round(total_mb,3)} mb")

        # Transfer params between time windows for init for s2s and curriculum purposes
        if config.transfer.s2s_transfer == True and config.transfer.s2s_pi_init == False:
            logger.info(f"About to restore model from checkpoint")
            if config.transfer.curriculum == True and idx == 0:
                ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_1")
            else:
                ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx))
            if os.path.exists(ckpt_path):
                    state = restore_checkpoint(model.state, ckpt_path, step=None) # Load latest checkpoint for tw

                    # Add an additional array embedding to every element
                    def add_array_embedding(x):
                        return jnp.array([x])
                    
                    params = {'params':None}
                    params['params'] = jax.tree_map(add_array_embedding, state.params['params'])
                    weights = jax.tree_map(add_array_embedding, state.weights)
                    model.state = model.state.replace(params=params, weights=weights)

        # Train model for the current time window
        model = train_one_window(config, workdir, model, samplers, idx, u, h)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params

            u0 = model.u0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star)
            # v0 = model.v0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)
            h0 = model.h0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star)

            del model, state, params

        # Reset if we use pi_init
        if idx == 0:
            config.transfer.s2s_pi_init = False # Turn off after first time window