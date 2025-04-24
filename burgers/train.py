import os
import time
import re

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
from jaxpi.samplers import UniformSampler, RADSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset, convert_config_to_dict, plot_rba_weights, plot_colloc_pts, plot_residuals, get_tree_size_mb


def train_one_window(config, workdir, model, samplers, idx, u_ref):
    # Initialize logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.BurgersEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    logger.info("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        if config.weighting.use_rad == True:
            if step % config.weighting.rad_update_every_steps == 0:
                batch, residuals, s0 = samplers['rad_sampler'](model.state)
        else:
            batch = next(samplers['res_sampler'])

        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch_log = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch_log, u_ref)

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
    # mode = "offline"
        
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config))
    logger.info(f"wandb initialized {mode}")

    # Get dataset
    u_ref, t_star, x_star = get_dataset(os.path.join("data",config.dataset))

    u0 = u_ref[0, :]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    t1 = t[-1] + (t[-1] - t[-2]) # * (1 + 0.01)  # cover the end point of each time window

    x0 = x_star[0]
    x1 = x_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Get NU (viscosity)
    REYNOLDS = int(re.search(r'Re(\d+)\.mat', config.dataset).group(1))
    U = 1 # characteristic velocity
    L = 2 # characteristic length
    NU = U*L/REYNOLDS
    config.nu = NU

    # Residual function that wraps the model's residual computation for rad
    residual_fn = lambda state, points: model.residuals(state, points)

    # Define residual sampler of collocation points
    samplers = {
        "res_sampler": iter(UniformSampler(dom, config.training.batch_size_per_device)),
        "rad_sampler": RADSampler(dom, config.training.batch_size_per_device, residual_fn, config.weighting.rad_k, config.weighting.rad_c),
    }
    
    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization (!!! might need to turn off for s2s!!!)
        if config.use_pi_init == True and config.transfer.curriculum == False:
            logger.info("Use physics-informed initialization...")

            model = models.Burgers(config, u0, t, x_star)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "linear_pde":
                # load data
                data = scipy.io.loadmat(os.path.join("data",config.dataset))
                # downsample the grid and data
                u = data["usol"][::10]
                t = data["t"].flatten()[::10]
                x = data["x"].flatten()

                tt, xx = jnp.meshgrid(t, x, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])

            elif config.pi_init_type == "initial_condition":
                t_init = t_star[::10]
                x = x_star
                u = u0

                tt, xx = jnp.meshgrid(t_init, x, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
                u = jnp.tile(u.flatten(), (t_init.shape[0], 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            coeffs, residuals, rank, s = jnp.linalg.lstsq(
                feat_matrix, u.flatten(), rcond=None
            )
            logger.info(f"least square residuals: {residuals}")

            config.arch.pi_init = coeffs.reshape(-1, 1)

            del model, state, params

        # Initialize model
        model = models.Burgers(config, u0, t, x_star)

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
        model = train_one_window(config, workdir, model, samplers, idx, u_ref)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            u0 = vmap(model.u_net, (None, None, 0))(
                params, t_star[num_time_steps], x_star
            )
            
            del model, state, params
     
    # return model
