from functools import partial
import time
import os
from absl import logging

from flax.training import checkpoints
import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from jax.tree_util import tree_map

import numpy as np
import scipy.io
import ml_collections
import wandb

import models
from jaxpi.utils import restore_checkpoint
from jaxpi.logging import Logger
from utils import get_dataset, plot_coordinates, get_bc_coords, plot_g_schedule, g_schedule_step, g_schedule_sigmoid

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import animation
from matplotlib.colors import Normalize


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize logger
    logger = Logger()

    # Get ref solution
    (h_ref, u_ref, v_ref, b_ref, t_ref, x_ref, y_ref, s_ref, g, manning) = get_dataset(config.dataset)

    # Remove last time step
    if config.training.num_time_windows > 1:
        h_ref = h_ref[:-1]
        u_ref = u_ref[:-1]
        v_ref = v_ref[:-1]
        b_ref = b_ref[:-1]
        s_ref = s_ref[:-1]
        t_ref = t_ref[:-1]

    # Nondimensionalization
    if config.nondim.nondimensionalize == True:
        # Nondimensionalization parameters        
        U_star = config.nondim.U_star   # characteristic velocity
        L_star = config.nondim.L_star   # characteristic length
        H_star = config.nondim.H_star   # characteristic height
        T_star = L_star / U_star        # characteristic time
        Froude_star = U_star / jnp.sqrt(g * config.nondim.H_star)

        # Nondimensionalize the flow field
        t_star = t_ref / T_star  # Non-dimensionalize time
        x_star = x_ref / L_star  # Non-dimensionalize x
        y_star = y_ref / L_star  # Non-dimensionalize y
        u_star = u_ref / U_star  # Non-dimensionalize velocity in x
        v_star = v_ref / U_star  # Non-dimensionalize velocity in y
        h_star = h_ref / H_star  # Non-dimensionalize height
        b_star = b_ref / H_star  # Non-dimensionalize bathymetry
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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

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
        
    # To restore model
    if config.use_pi_init == True and config.transfer.curriculum == False:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    # Initialize model
    model = models.SWE2D_NC(config, u0, v0, h0, t, x_star, y_star, bc_coords, Froude_star, g, g_values, manning)

    u_pred_list = []
    v_pred_list = []
    h_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        h = h_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Restore the checkpoint
        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_u_error, l2_v_error, l2_h_error = model.compute_l2_error(params, t, x_star, y_star, u, v, h)
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))
        logging.info("Time window: {}, h error: {:.3e}".format(idx + 1, l2_h_error))

        # Compute window prediction
        u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)
        h_pred = model.h_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)
        h_pred_list.append(h_pred)

    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)
    h_pred = jnp.concatenate(h_pred_list, axis=0)
    
    u_error = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    v_error = jnp.linalg.norm(v_pred - v_star) / jnp.linalg.norm(v_star)
    h_error = jnp.linalg.norm(h_pred - h_star) / jnp.linalg.norm(h_star)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))
    logging.info("L2 error of the full prediction of h: {:.3e}".format(h_error))

    u_mae = jnp.mean(jnp.abs(u_pred - u_star)) / jnp.mean(jnp.abs(u_star))
    v_mae = jnp.mean(jnp.abs(v_pred - v_star)) / jnp.mean(jnp.abs(v_star))
    h_mae = jnp.mean(jnp.abs(h_pred - h_star)) / jnp.mean(jnp.abs(h_star))

    logging.info("MAE of the full prediction of u: {:.3e}".format(u_mae))
    logging.info("MAE of the full prediction of v: {:.3e}".format(v_mae))
    logging.info("MAE of the full prediction of h: {:.3e}".format(h_mae))

    # Dimensionalize flow field
    if config.nondim.nondimensionalize == True:
        u_pred = u_pred * U_star
        v_pred = v_pred * U_star
        h_pred = h_pred * H_star


    # -------------- Compute mass loss --------------

    # Calculate grid spacing
    dx = x_ref[1] - x_ref[0]
    dy = y_ref[1] - y_ref[0]

    # Compute initial mass from reference solution at t=0
    mass_initial = np.sum(h_ref[0]) * dx * dy

    # Compute mass for each time step in the predicted solution
    mass_pred = np.sum(h_pred, axis=(1, 2)) * dx * dy

    # Calculate relative mass loss compared to initial mass
    mass_loss = np.abs((mass_pred - mass_initial) / mass_initial)

    # Compute the mean mass loss across all time steps
    mean_mass_loss = np.mean(mass_loss)

    # Calculate percent values
    absolute_pct_error = mass_loss * 100
    mape = mean_mass_loss * 100

    logging.info(f"Mean absolute percent mass error: {mape:.4e}")
    logging.info(f"Absolute percent mass error: {absolute_pct_error}")

    # Focus y-axis around relevant data scale
    y_min = max(0, absolute_pct_error.min() - 1)
    y_max = absolute_pct_error.max() + 1

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_ref, absolute_pct_error, label="PINN Absolute Percent Error", color='blue', linewidth=2)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Absolute Percent Error (%)", fontsize=12)
    plt.title("PINN absolute percent mass error over time", fontsize=14)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'swe_dam_mass.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    wandb.log({"SWE mass mape": wandb.Image(save_path)})


    # -------------- Plot GIF --------------

    # Create the gif
    fig, axes = plt.subplots(5, 3, figsize=(20, 25))
    # plt.tight_layout()

    # Titles for subplots
    titles = ["$h$", "$s$", "$b$", "$u$", "$v$"]
    column_titles = ["Predicted", "Reference", "Absolute Error"]

    # Create the meshgrid for pcolormesh (assuming xx and yy are 1D arrays defining the grid)
    Nt, Nx, Ny = u_star.shape
    dt = t_ref[1] - t_ref[0]
    xx, yy = jnp.meshgrid(x_ref, y_ref, indexing="ij")

    # # Calculate bathymetry
    # def bathymetry(x, y):
    #     r2 = (x - 1.0)**2 + (y - 0.5)**2
    #     return 0.8 * np.exp(-10 * r2)
    # b = bathymetry(xx, yy)
    # b = b[np.newaxis, :, :]
    # b = np.repeat(b, h_pred.shape[0], axis=0)

    # Reshape preds and refs
    # u_pred = u_pred.reshape(Nt * config.training.num_time_windows, Nx, Ny)
    # v_pred = v_pred.reshape(Nt * config.training.num_time_windows, Nx, Ny)
    # h_pred = h_pred.reshape(Nt * config.training.num_time_windows, Nx, Ny)

    # Calculate the surface height for predictions and reference
    s_pred = h_pred + b_ref
    s_ref = h_ref + b_ref

    # Determine the global colorbar limits
    fields = [(h_pred, h_ref), (s_pred, s_ref), (b_ref, b_ref), (u_pred, u_ref), (v_pred, v_ref)]
    errors = [np.abs(pred - ref) for pred, ref in fields]

    color_limits = [
        (field[0].min(), field[0].max(), field[1].min(), field[1].max(), error.max())
        for field, error in zip(fields, errors)
    ]

    # Make colorbar scale uniform
    for i in range(len(color_limits)):
        field_min = min(color_limits[i][0], color_limits[i][2])
        field_max = max(color_limits[i][1], color_limits[i][3])
        
        # Replace the tuple with a new one
        color_limits[i] = (field_min, field_max, field_min, field_max, color_limits[i][4])

    # Create the initial pcolormesh plots and colorbars
    pcms = []
    for row, (field, (vmin_pred, vmax_pred, vmin_ref, vmax_ref, vmax_error), title) in enumerate(zip(fields, color_limits, titles)):
        for col, data, vmin, vmax, col_title in zip(
            range(3), 
            [field[0][0], field[1][0], np.abs(field[0][0] - field[1][0])], 
            [vmin_pred, vmin_ref, 0], 
            [vmax_pred, vmax_ref, vmax_error], 
            column_titles
        ):
            ax = axes[row, col]
            pcm = ax.pcolormesh(xx, yy, data, cmap="jet", shading="auto", vmin=vmin, vmax=vmax)
            pcms.append(pcm)
            ax.set_title(f"{title} ({col_title})", fontsize=16)
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            fig.colorbar(pcm, ax=ax)

    # Function to update the plots for animation
    def update(frame):
        for row, field in enumerate(fields):
            pred, ref = field
            error = np.abs(pred[frame] - ref[frame])
            for col, data in enumerate([pred[frame], ref[frame], error]):
                pcm_index = row * 3 + col
                pcms[pcm_index].set_array(data.ravel())  # Update the pcolormesh data
                axes[row, col].set_title(f"{titles[row]} ({column_titles[col]}) at time {t_ref[frame]:.2f}s", fontsize=16)
        return pcms

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=u_pred.shape[0], blit=False)

    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save as GIF
    file_name = "swe_fields.gif"
    save_path = os.path.join(save_dir, file_name)
    ani.save(save_path, writer=animation.PillowWriter(fps=10))
    print(f"Animation saved at {save_path}")

    wandb.log({f"SWE u,v,h fields": wandb.Video(os.path.join(save_dir, file_name))})

    # Plot evolution of g
    if config.training.g_schedule != None:
        plot_g_schedule(config, workdir, config.training.g_schedule, g)


    # -------------- Plot slices at y=0 --------------
    for num_slices in [3,5]:
        # Find the index where y=0
        y_idx = int(len(y_star)/2)

        # Select 5 evenly spaced time indices
        num_time_steps = len(t_ref)
        selected_indices = np.linspace(0, num_time_steps-1, num_slices, dtype=int)
        selected_times = t_ref[selected_indices]

        # Create figure with two subplots
        fig, (ax_s, ax_u) = plt.subplots(1, 2, figsize=(16, 6))

        # Set up colormap for time values
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=selected_times.min(), vmax=selected_times.max())

        s_ref = np.array(s_ref)
        s_pred = np.array(s_pred)
        u_ref = np.array(u_ref)
        u_pred = np.array(u_pred)

        # Plot slices for each selected time
        for idx, t in zip(selected_indices, selected_times):
            color = cmap(norm(t))
            
            # Plot S field
            ax_s.plot(x_ref, s_ref[idx, :, y_idx], 
                    color=color, linestyle='-', label='Reference' if idx == selected_indices[0] else "")
            ax_s.plot(x_ref, s_pred[idx, :, y_idx], 
                    color=color, linestyle='--', label='Predicted' if idx == selected_indices[0] else "")
            
            # Plot U field
            ax_u.plot(x_ref, u_ref[idx, :, y_idx], 
                    color=color, linestyle='-', label='Reference' if idx == selected_indices[0] else "")
            ax_u.plot(x_ref, u_pred[idx, :, y_idx], 
                    color=color, linestyle='--', label='Predicted' if idx == selected_indices[0] else "")

        # Add labels and titles
        ax_s.set_title('Surface Height (s)')
        ax_s.set_xlabel('x')
        ax_s.set_ylabel('s')
        ax_s.grid(True)

        ax_u.set_title('Velocity (u)')
        ax_u.set_xlabel('x')
        ax_u.set_ylabel('u')
        ax_u.grid(True)

        # Create colorbar for time values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar = fig.colorbar(sm, ax=[ax_s, ax_u], orientation='vertical', pad=0.02)
        cbar.set_label('Time (s)')

        # Add legend
        handles, labels = ax_s.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc='lower center', ncol=2)

        # Add global title with selected times
        time_str = ", ".join([f"{t:.2f}" for t in selected_times])
        fig.suptitle(f"Radial dam break: S and U cross sections at times [{time_str}] s", fontsize=14, y=1.02)

        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust layout to make room for legend

        # Save the plot
        save_path = os.path.join(workdir, "figures", config.wandb.name, f"time_slices_{num_slices}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Time slice plot saved at {save_path}")
        wandb.log({f"SWE Time Slices {num_slices}": wandb.Image(save_path)})


    # -------------- Compute wave speeds --------------
    def compute_wave_speeds_2d(u, v, h, g):
        c_x_minus = u - np.sqrt(g * h)
        c_x_plus  = u + np.sqrt(g * h)
        c_y_minus = v - np.sqrt(g * h)
        c_y_plus  = v + np.sqrt(g * h)
        return c_x_minus, c_x_plus, c_y_minus, c_y_plus

    # Predicted and reference wave speeds
    cxm_pred, cxp_pred, cym_pred, cyp_pred = compute_wave_speeds_2d(u_pred, v_pred, h_pred, config.setup.g)
    # cxm_pred, cxp_pred, cym_pred, cyp_pred = compute_wave_speeds_2d(jnp.ones_like(u_pred), jnp.ones_like(v_pred), jnp.ones_like(h_pred), config.setup.g)
    cxm_ref, cxp_ref, cym_ref, cyp_ref = compute_wave_speeds_2d(u_ref, v_ref, h_ref, config.setup.g)

    # wave_cxp_pred and wave_cxp_star should be shaped (Nt, Nx, Ny)
    wave_cxp_error = np.mean(np.abs(cxp_pred - cxp_ref), axis=(1, 2))
    wave_cxm_error = np.mean(np.abs(cxm_pred - cxm_ref), axis=(1, 2))
    wave_cyp_error = np.mean(np.abs(cyp_pred - cyp_ref), axis=(1, 2))
    wave_cym_error = np.mean(np.abs(cym_pred - cym_ref), axis=(1, 2))

    # MAE
    mae_wave_cxp = np.mean(wave_cxp_error)
    mae_wave_cxm = np.mean(wave_cxm_error)
    mae_wave_cyp = np.mean(wave_cyp_error)
    mae_wave_cym = np.mean(wave_cym_error)

    # Logging
    logging.info(f"Right wave x (c+_x) MAE: {mae_wave_cxp}")
    logging.info(f"Left wave x (c-_x) MAE: {mae_wave_cxm}")
    logging.info(f"Upward wave y (c+_y) MAE: {mae_wave_cyp}")
    logging.info(f"Downward wave y (c-_y) MAE: {mae_wave_cym}")

    plt.figure(figsize=(10, 6))

    plt.plot(t_ref, wave_cxp_error, label="Right-going wave (c⁺ₓ)", color='blue', linewidth=2)
    plt.plot(t_ref, wave_cxm_error, label="Left-going wave (c⁻ₓ)", color='red', linewidth=2)
    plt.plot(t_ref, wave_cyp_error, label="Upward-going wave (c⁺ᵧ)", color='green', linewidth=2)
    plt.plot(t_ref, wave_cym_error, label="Downward-going wave (c⁻ᵧ)", color='orange', linewidth=2)

    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Mean Absolute Wave Speed Error", fontsize=12)
    plt.title("2D SWE Wave Speed Error Over Time", fontsize=14)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)

    file_name = f'swe2d_wave_speed_error.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    wandb.log({"SWE 2D wave speed error": wandb.Image(save_path)})