import os, sys
import time
import re
import gc

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import scipy.io

import ml_collections
from absl import logging
import wandb
import platform

from jaxpi.archs import PeriodEmbs, Embedding
from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset, convert_config_to_dict, plot_collocation_points, get_bc_coords_values


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize logger
    logger = Logger()

    # Find out if running on pc for dubugging or on HPC without internet access
    if 'microsoft' in platform.uname().release.lower():
        mode = "online"
    else:
        mode = "offline"
        
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config), tags=wandb_config.tag)
    logger.info(f"wandb initialized {mode}")

    # Get dataset
    u_ref, v_ref, t_star, x_star, y_star, NU = get_dataset(os.path.join("data",config.dataset))
    # u_ref = u_ref[::2,::2,::2]
    # v_ref = v_ref[::2,::2,::2]
    # t_star = t_star[::2]
    # x_star = x_star[::2]
    # y_star = y_star[::2]

    u0 = u_ref[0, :]
    v0 = v_ref[0, :]

    t0 = t_star[0]
    t1 = t_star[-1]

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Define bc coords
    bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    # Get NU (viscosity)
    REYNOLDS = int(re.search(r'Re(\d+)\.mat', config.dataset).group(1))
    L = 1 # characteristic length
    U = NU * REYNOLDS / L # characteristic velocity
    config.nu = NU
    config.reynolds = REYNOLDS

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))
    
    # Physics-informed initialization
    if config.use_pi_init == True: # and config.transfer.curriculum == False:
        logger.info("Use physics-informed initialization...")

        model = models.Burgers2d(config, u0, v0, t_star, x_star, y_star, bc_coords, bc_values)
        state = jax.device_get(tree_map(lambda x: x[0], model.state))
        params = state.params

        # Initialization data source
        if config.pi_init_type == "linear_pde":
            # load data
            data = scipy.io.loadmat(os.path.join("data",config.dataset))
            # downsample the grid and data
            u = data["usol"][::10]
            v = data["vsol"][::10]
            t = data["t"].flatten()[::10]
            x = data["x"].flatten()
            y = data["y"].flatten()

            tt, xx, yy = jnp.meshgrid(t, x, y, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])


        elif config.pi_init_type == "initial_condition":
            t = t_star[::5]
            x = x_star
            y = y_star
            u = u0
            v = v0

            tt, xx, yy = jnp.meshgrid(t, x, y, indexing="ij")
            inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])
            u = jnp.tile(u, (t.shape[0], 1, 1))
            v = jnp.tile(v, (t.shape[0], 1, 1))

        # Compute feature matrix for both 'u' and 'v'
        feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

        u_coeffs, u_res, rank, s = jnp.linalg.lstsq(feat_matrix, u.flatten(), rcond=None)
        v_coeffs, v_res, rank, s = jnp.linalg.lstsq(feat_matrix, v.flatten(), rcond=None)

        logger.info(f"least square u residuals: {u_res}")
        logger.info(f"least square v residuals: {v_res}")

        coeffs = jnp.vstack([u_coeffs, v_coeffs]).T
        config.arch.pi_init = coeffs
        
        del model, state, params # tt, xx, yy, _, feat_matrix, rank, u_coeffs, v_coeffs, u_res, v_res, s

    # Initialize model
    model = models.Burgers2d(config, u0, v0, t_star, x_star, y_star, bc_coords, bc_values)

    # Update initial weights and params if transfer learning
    if config.transfer.curriculum == True:
        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt")
        if os.path.exists(os.path.join(ckpt_path,f"checkpoint_{config.transfer.curri_step}")):
            state = restore_checkpoint(model.state, ckpt_path, step=config.transfer.curri_step)

            # Add an additional array embedding to every element
            def add_array_embedding(x):
                return jnp.array([x]) #if not isinstance(x, jnp.ndarray) or x.ndim == 0 else x
            
            params = {'params':None}
            params['params'] = jax.tree_map(add_array_embedding, state.params['params'])
            weights = jax.tree_map(add_array_embedding, state.weights)
            model.state = model.state.replace(params=params, weights=weights)

    # Initialize evaluator
    evaluator = models.Burgers2dEvaluator(config, model)

    logger.info("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        # if step == 0:
        #     save_dir = os.path.join(workdir,'figures',wandb_config.name)
        #     if not os.path.isdir(save_dir):
        #         os.makedirs(save_dir)
        #     plot_collocation_points(np.array(batch).reshape(-1,3),save_dir)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref, v_ref)

                # Get global step if using curriculum training
                if config.transfer.curriculum == True:
                    step_logged = step + config.logging.global_step
                    
                    # Skip last step except for last dataset in curriculum
                    if (step+1) != config.training.max_steps:
                        wandb.log(log_dict, step_logged)
                        end_time = time.time()
                        logger.log_iter(step_logged, start_time, end_time, log_dict)

                # Without curriculum
                else:
                    wandb.log(log_dict, step)
                    end_time = time.time()
                    logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)
    
    # Delete the model to free GPU memory
    logger.info(f"Model size: {sys.getsizeof(model)} bytes")
    del model, state
    del u0, v0, bc_coords, bc_values #, batch, inputs
    del u_ref, v_ref, t_star, x_star, y_star, NU #, t, x, y, u, v
    gc.collect()
    logger.info("Deleted model and collected garbage from memory")

    # return model


def train_one_window(config, workdir, model, res_sampler, idx, u_ref, v_ref):
    # Initialize logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.Burgers2dEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    logger.info("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        # if step == 0:
        #     save_dir = os.path.join(workdir,'figures',config.wandb.name)
        #     if not os.path.isdir(save_dir):
        #         os.makedirs(save_dir)
        #     plot_collocation_points(np.array(batch).reshape(-1,2),save_dir)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref, v_ref)

                # Get global step if using curriculum training
                if config.transfer.curriculum == True:
                    step_logged = step + config.logging.global_step
                    
                    # Skip last step except for last dataset in curriculum
                    if (step+1) != config.training.max_steps:
                        wandb.log(log_dict, step_logged + step_offset)
                        end_time = time.time()
                        logger.log_iter(step_logged, start_time, end_time, log_dict)

                # Without curriculum
                else:
                    wandb.log(log_dict, step + step_offset)
                    end_time = time.time()
                    logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model

def train_and_evaluate_s2s(config: ml_collections.ConfigDict, workdir: str):
    """Train and evaluate the model using sequence 2 sequence learning from Krishnapriyan et al. (2021)"""
    # Initialize logger
    logger = Logger()

    # Find out if running on pc for dubugging or on HPC without internet access
    if 'microsoft' in platform.uname().release.lower():
        mode = "online"
    else:
        mode = "offline"
        
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config))
    logger.info(f"wandb initialized {mode}")

    # Get dataset
    u_ref, v_ref, t_star, x_star, y_star, NU = get_dataset(os.path.join("data",config.dataset))
    # u_ref = u_ref[::20,::20,::20]
    # v_ref = v_ref[::20,::20,::20]
    # t_star = t_star[::20]
    # x_star = x_star[::20]
    # y_star = y_star[::20]
    u0 = u_ref[0, :, :]
    v0 = v_ref[0, :, :]

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    t1 = t[-1] + (t[-1] - t[-2]) # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Define bc coords
    bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    # Get NU (viscosity)
    REYNOLDS = int(re.search(r'Re(\d+)\.mat', config.dataset).group(1))
    L = 1 # characteristic length
    U = NU * REYNOLDS / L # characteristic velocity
    config.nu = NU
    config.reynolds = REYNOLDS

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization (!!! might need to turn off for s2s!!!)
        if config.use_pi_init == True and config.transfer.curriculum == False:
            logger.info("Use physics-informed initialization...")

            # model = models.Burgers(config, u0, t_star, x_star)
            model = models.Burgers2d(config, u0, v0, t, x_star, y_star, bc_coords, bc_values)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "linear_pde":
                # load data
                data = scipy.io.loadmat(os.path.join("data",config.dataset))
                # downsample the grid and data
                u = data["usol"][::10]
                v = data["vsol"][::10]
                t = data["t"].flatten()[::10]
                x = data["x"].flatten()
                y = data["y"].flatten()

                t_scaled = t / t[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])

            elif config.pi_init_type == "initial_condition":
                t = t_star[::10]
                x = x_star
                y = y_star
                u = u0
                v = v0

                t_scaled = t / t[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])
                u = jnp.tile(u, (t_scaled.shape[0], 1, 1))
                v = jnp.tile(v, (t_scaled.shape[0], 1, 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            u_coeffs, u_res, rank, s = jnp.linalg.lstsq(feat_matrix, u.flatten(), rcond=None)
            v_coeffs, v_res, rank, s = jnp.linalg.lstsq(feat_matrix, v.flatten(), rcond=None)

            logger.info(f"least square u residuals: {u_res}")
            logger.info(f"least square v residuals: {v_res}")

            coeffs = jnp.vstack([u_coeffs, v_coeffs]).T
            config.arch.pi_init = coeffs

            del model, state, params #, tt, xx, yy

        # Initialize model
        model = models.Burgers2d(config, u0, v0, t, x_star, y_star, bc_coords, bc_values)

        if idx == 0:
            # Update initial weights and params if transfer learning
            if config.transfer.curriculum == True:
                ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt")
                if os.path.exists(os.path.join(ckpt_path,f"checkpoint_{config.transfer.curri_step}")):
                    state = restore_checkpoint(model.state, ckpt_path, step=config.transfer.curri_step)

                    # Add an additional array embedding to every element
                    def add_array_embedding(x):
                        return jnp.array([x]) #if not isinstance(x, jnp.ndarray) or x.ndim == 0 else x
                    
                    params = {'params':None}
                    params['params'] = jax.tree_map(add_array_embedding, state.params['params'])
                    weights = jax.tree_map(add_array_embedding, state.weights)
                    model.state = model.state.replace(params=params, weights=weights)

        # Train model for the current time window
        model = train_one_window(config, workdir, model, res_sampler, idx, u, v)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            # u0 = vmap(model.u_net, (None, None, 0))(params, t_star[num_time_steps], x_star)
            u0 = model.u0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            v0 = model.v0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            
            del model, state, params

