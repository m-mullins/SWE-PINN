import functools
from functools import partial
import time
import os

from absl import logging

import jax

import jax.numpy as jnp
from jax import random, vmap, pmap, local_device_count
from jax.tree_util import tree_map

import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import ml_collections

import wandb
import platform

import models

from jaxpi.samplers import UniformSampler, BaseSampler, SpaceSampler, TimeSpaceSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

# from utils import get_dataset, get_fine_mesh, parabolic_inflow
from utils import get_dataset, convert_config_to_dict, get_bc_coords, g_schedule_step, g_schedule_sigmoid, count_params


# class ICSampler(SpaceSampler):
#     def __init__(self, u, v, h, coords, batch_size, rng_key=random.PRNGKey(1234)):
#         super().__init__(coords, batch_size, rng_key)

#         self.u = u
#         self.v = v
#         self.h = h

#     @partial(pmap, static_broadcasted_argnums=(0,))
#     def data_generation(self, key):
#         "Generates data containing batch_size samples"
#         idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))

#         coords_batch = self.coords[idx, :]

#         u_batch = self.u[idx]
#         v_batch = self.v[idx]
#         h_batch = self.h[idx]

#         batch = (coords_batch, u_batch, v_batch, h_batch)

#         return batch


# class ResSampler(BaseSampler):
#     def __init__(
#         self,
#         temporal_dom,
#         coarse_coords,
#         fine_coords,
#         batch_size,
#         rng_key=random.PRNGKey(1234),
#     ):
#         super().__init__(batch_size, rng_key)

#         self.temporal_dom = temporal_dom

#         self.coarse_coords = coarse_coords
#         self.fine_coords = fine_coords

#     @partial(pmap, static_broadcasted_argnums=(0,))
#     def data_generation(self, key):
#         "Generates data containing batch_size samples"
#         subkeys = random.split(key, 4)

#         temporal_batch = random.uniform(
#             subkeys[0],
#             shape=(2 * self.batch_size, 1),
#             minval=self.temporal_dom[0],
#             maxval=self.temporal_dom[1],
#         )

#         coarse_idx = random.choice(
#             subkeys[1],
#             self.coarse_coords.shape[0],
#             shape=(self.batch_size,),
#             replace=True,
#         )

#         fine_idx = random.choice(
#             subkeys[2],
#             self.fine_coords.shape[0],
#             shape=(self.batch_size,),
#             replace=True,
#         )

#         coarse_spatial_batch = self.coarse_coords[coarse_idx, :]
#         fine_spatial_batch = self.fine_coords[fine_idx, :]
#         spatial_batch = jnp.vstack([coarse_spatial_batch, fine_spatial_batch])
#         spatial_batch = random.permutation(
#             subkeys[3], spatial_batch
#         )  # mix the coarse and fine coordinates

#         batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)

#         return batch


def train_one_window(config, workdir, model, res_sampler, idx, u_ref, v_ref, h_ref):
    # Initialize evaluator
    evaluator = models.SWEEvaluator(config, model)

    # Initialize logger
    logger = Logger()

    step_offset = idx * config.training.max_steps

    # jit warm up
    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        # Update weights if necessary
        if config.weighting.scheme in ["grad_norm"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref, v_ref, h_ref)

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
    logger.info(f"Config: {config.wandb.name}")
    logger.info(f"Dataset: {config.dataset}")

    # Get ref solution
    (h_ref, u_ref, v_ref, b_ref, t_ref, x_ref, y_ref, s_ref, g, manning) = get_dataset(config.dataset)
    logger.info(f"g = {g}")

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
        y_star = y_ref / L_star  # Non-dimensionalize y
        u_star = u_ref / U_star  # Non-dimensionalize velocity in x
        v_star = v_ref / U_star  # Non-dimensionalize velocity in y
        h_star = h_ref / H_star  # Non-dimensionalize height
        b_star = b_ref / H_star  # Non-dimensionalize bathymetry
        # logger.info(f"u_star.max {u_star.max()}")
        # logger.info(f"u_star.min {u_star.min()}")
        # logger.info(f"v_star.max {v_star.max()}")
        # logger.info(f"v_star.min {v_star.min()}")
        # logger.info(f"h_star.max {h_star.max()}")
        # logger.info(f"h_star.min {h_star.min()}")
    else:
        t_star = t_ref  # Non-dimensionalize time
        x_star = x_ref  # Non-dimensionalize x
        y_star = y_ref  # Non-dimensionalize y
        u_star = u_ref  # Non-dimensionalize velocity in x
        v_star = v_ref  # Non-dimensionalize velocity in y
        h_star = h_ref  # Non-dimensionalize height
        b_star = b_ref  # Non-dimensionalize bathymetry
        Froude_star = 1 / jnp.sqrt(g)
        logger.info(f"Froude* = {Froude_star}")

    u0 = u_star[0, :, :]
    v0 = v_star[0, :, :]
    h0 = h_star[0, :, :]

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    dt = round(t[-1] - t[-2],6)
    t1 = t[-1] + dt # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Calculate numerical viscosity
    # if config.nondim.nondimensionalize == True:
    #     config.nondim.visc = U_star**2 * dt / 2
    # else:
    #     config.nondim.visc = dt / 2
    if config.setup.use_visc == True:
        logger.info(f"Numerical viscosity = {config.nondim.visc}")

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    keys = random.split(random.PRNGKey(0), config.training.num_time_windows)

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

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
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        h = h_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization
        if config.use_pi_init == True and config.transfer.s2s_pi_init == True:
            logger.info("Use physics-informed initialization...")

            model = models.SWE2D_NC(config, u0, v0, h0, t, x_star, y_star, bc_coords, Froude_star, g, g_values, manning)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "initial_condition":
                t_init = t_star[::2]
                downsample = 1
                x = x_star[::downsample]
                y = y_star[::downsample]

                u_init = u0[::downsample, ::downsample]
                v_init = v0[::downsample, ::downsample]
                h_init = h0[::downsample, ::downsample]

                t_scaled = t_init / t_init[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])

                u_init = jnp.tile(u_init, (t_scaled.shape[0], 1, 1))
                v_init = jnp.tile(v_init, (t_scaled.shape[0], 1, 1))
                h_init = jnp.tile(h_init, (t_scaled.shape[0], 1, 1))
                h_init_max = h_init.max()
                h_init_min = h_init.min()

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            u_coeffs, u_res, rank, s = jnp.linalg.lstsq(feat_matrix, u_init.flatten(), rcond=None)
            v_coeffs, v_res, rank, s = jnp.linalg.lstsq(feat_matrix, v_init.flatten(), rcond=None)
            h_coeffs, h_res, rank, s = jnp.linalg.lstsq(feat_matrix, h_init.flatten(), rcond=None)

            logger.info(f"least square u residuals: {u_res}")
            logger.info(f"least square v residuals: {v_res}")
            logger.info(f"least square h residuals: {h_res}")

            coeffs = jnp.vstack([u_coeffs, v_coeffs, h_coeffs]).T
            config.arch.pi_init = coeffs

            del model, state, params

        # Initialize model
        model = models.SWE2D_NC(config, u0, v0, h0, t, x_star, y_star, bc_coords, Froude_star, g, g_values, manning)

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
        model = train_one_window(config, workdir, model, res_sampler, idx, u, v, h)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params

            u0 = model.u0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)
            v0 = model.v0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)
            h0 = model.h0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)

            del model, state, params

        # Reset if we use pi_init
        if idx == 0:
            config.transfer.s2s_pi_init = False # Turn off after first time window