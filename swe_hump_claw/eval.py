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
from utils import get_dataset, get_bc_coords, plot_g_schedule, g_schedule_step, g_schedule_sigmoid

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import animation
from matplotlib.colors import Normalize


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize logger
    logger = Logger()

    # Get ref solution
    (h_ref, u_ref, hu_ref, b_ref, t_ref, x_ref, s_ref, g, manning) = get_dataset(config.dataset)

    # Remove last time step
    if config.training.num_time_windows > 1:
        h_ref = h_ref[:-1]
        u_ref = u_ref[:-1]
        hu_ref = hu_ref[:-1]
        # v_ref = v_ref[:-1]
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
        # y_star = y_ref / L_star  # Non-dimensionalize y
        u_star = u_ref / U_star  # Non-dimensionalize velocity in x
        hu_star = hu_ref / (H_star * U_star) # Non-dimensionalize momentum in x
        # v_star = v_ref / U_star  # Non-dimensionalize velocity in y
        h_star = h_ref / H_star  # Non-dimensionalize height
        b_star = b_ref / H_star  # Non-dimensionalize bathymetry
    else:
        t_star = t_ref  # Non-dimensionalize time
        x_star = x_ref  # Non-dimensionalize x
        # y_star = y_ref  # Non-dimensionalize y
        u_star = u_ref  # Non-dimensionalize velocity in x
        hu_star = hu_ref # Non-dimensionalize momentum in x
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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t)

    # # Inflow boundary conditions
    # inflow_fn = lambda y: parabolic_inflow(y * L_star, config.setup.U_max)

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
    model = models.SWE2D_NC(config, u0, h0, t, x_star, bc_coords, Froude_star, g, g_values, manning)

    u_pred_list = []
    # v_pred_list = []
    h_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        # v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        h = h_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Restore the checkpoint
        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_u_error, l2_h_error = model.compute_l2_error(params, t, x_star, u, h)
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        # logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))
        logging.info("Time window: {}, h error: {:.3e}".format(idx + 1, l2_h_error))

        # Compute window prediction
        u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
        # v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)
        h_pred = model.h_pred_fn(params, model.t_star, model.x_star)

        u_pred_list.append(u_pred)
        # v_pred_list.append(v_pred)
        h_pred_list.append(h_pred)

    u_pred = jnp.concatenate(u_pred_list, axis=0)
    h_pred = jnp.concatenate(h_pred_list, axis=0)
    hu_pred = h_pred * u_pred
    
    u_error = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    h_error = jnp.linalg.norm(h_pred - h_star) / jnp.linalg.norm(h_star)
    hu_error = jnp.linalg.norm(hu_pred - hu_star) / jnp.linalg.norm(hu_star)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of h: {:.3e}".format(h_error))
    logging.info("L2 error of the full prediction of hu: {:.3e}".format(hu_error))

    u_mae = jnp.mean(jnp.abs(u_pred - u_star)) / jnp.mean(jnp.abs(u_star))
    h_mae = jnp.mean(jnp.abs(h_pred - h_star)) / jnp.mean(jnp.abs(h_star))
    hu_mae = jnp.mean(jnp.abs(hu_pred - hu_star)) / jnp.mean(jnp.abs(hu_star))

    logging.info("MAE of the full prediction of u: {:.3e}".format(u_mae))
    logging.info("MAE of the full prediction of h: {:.3e}".format(h_mae))
    logging.info("MAE of the full prediction of hu: {:.3e}".format(hu_mae))

    # Dimensionalize flow field
    if config.nondim.nondimensionalize == True:
        u_pred = u_pred * U_star
        h_pred = h_pred * H_star
        hu_pred = hu_pred * (H_star * U_star)


    # -------------- Compute mass loss --------------

    # Calculate grid spacing
    dx = x_ref[1] - x_ref[0]

    # Compute initial mass from reference solution at t=0
    mass_initial = np.sum(h_ref[0]) * dx

    # Compute mass for each time step in the predicted solution
    mass_pred = np.sum(h_pred, axis=1) * dx  # Sum over x dimension

    # Calculate absolute percent mass loss
    absolute_pct_error = np.abs((mass_pred - mass_initial) / mass_initial) * 100

    # Compute the mean absolute percent mass loss across all time steps
    mape = np.mean(absolute_pct_error)

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

    # Save png
    file_name = f'swe_hump_mass.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    wandb.log({"SWE mass mape": wandb.Image(save_path)})


    # -------------- Plot space-time --------------

    # Create time-space grid
    tt, xx = np.meshgrid(t_ref, x_ref, indexing='ij')  # tt shape (Nt, Nx), xx shape (Nt, Nx)
    
    # Calculate surface height
    s_pred = h_pred + b_ref[:]  # Add bathymetry
    s_ref = h_ref + b_ref[:]
    
    # Titles setup
    titles = ["$h$", "$s$", "$b$", "$hu$", "$u$"]
    column_titles = ["Predicted", "Reference", "Absolute Error"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(len(titles), len(column_titles), figsize=(20, 16))
    plt.tight_layout(pad=4.0)
    
    # Prepare data fields
    # b_ref_tiled = np.tile(b_ref, (len(t_star), 1))  # Repeat bathymetry over time
    fields = [
        (h_pred, h_ref),
        (s_pred, s_ref),
        (b_ref, b_ref),
        (hu_pred, hu_ref),
        (u_pred, u_ref)
    ]
    
    # Determine color limits
    color_limits = []
    for pred, ref in fields:
        combined = np.concatenate([pred, ref])
        umin = combined.min()
        umax = combined.max()
        error_max = np.abs(pred - ref).max()
        color_limits.append((umin, umax, error_max))
    
    # Create plots
    for row in range(len(titles)):
        for col in range(3):
            ax = axes[row, col]
            data = fields[row][0] if col == 0 else fields[row][1] if col == 1 else np.abs(fields[row][0] - fields[row][1])
            
            # Get appropriate color limits
            if col < 2:
                umin, umax, _ = color_limits[row]
                cmap = 'jet'
            else:
                umin, umax = 0, color_limits[row][2]
                cmap = 'jet'
            
            # Plot space-time diagram
            pcm = ax.pcolormesh(xx, tt, data, shading='auto', cmap=cmap, vmin=umin, vmax=umax)
            fig.colorbar(pcm, ax=ax)
            
            # Formatting
            ax.set_title(f"{titles[row]} ({column_titles[col]})")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("Time [s]")
    
    # Save results
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "space_time_results.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    
    # Log to wandb
    wandb.log({"Space-Time Results": wandb.Image(file_path)})
    
    print(f"Space-time plots saved to {file_path}")


    # -------------- Plot time slices --------------

    for num_slices in [3,5]:
        # Select 5 evenly spaced time indices
        num_time_steps = len(t_ref)
        selected_indices = np.linspace(0, num_time_steps-1, num_slices, dtype=int)
        selected_times = t_ref[selected_indices]

        # Create figure with two subplots
        fig, (ax_s, ax_u, ax_hu) = plt.subplots(1, 3, figsize=(24, 6))

        # Set up colormap for time values
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=selected_times.min(), vmax=selected_times.max())

        # Convert to numpy arrays if needed
        s_ref = np.array(s_ref)
        s_pred = np.array(s_pred)
        u_ref = np.array(u_ref)
        u_pred = np.array(u_pred)

        # Plot slices for each selected time
        for idx, t in zip(selected_indices, selected_times):
            color = cmap(norm(t))
            
            # Plot s field (1D slice)
            ax_s.plot(x_ref, s_ref[idx, :], 
                    color=color, linestyle='-', label='Reference' if idx == selected_indices[0] else "")
            ax_s.plot(x_ref, s_pred[idx, :], 
                    color=color, linestyle='--', label='Predicted' if idx == selected_indices[0] else "")
            ax_s.plot(x_ref, b_ref[idx, :], 
                    color="black", linestyle=':', label='Bathymetry' if idx == selected_indices[0] else "")

            # Plot s field (1D slice)
            ax_u.plot(x_ref, u_ref[idx, :], 
                    color=color, linestyle='-', label='Reference' if idx == selected_indices[0] else "")
            ax_u.plot(x_ref, u_pred[idx, :], 
                    color=color, linestyle='--', label='Predicted' if idx == selected_indices[0] else "")
            
            # Plot hu field (1D slice)
            ax_hu.plot(x_ref, hu_ref[idx, :], 
                    color=color, linestyle='-', label='Reference' if idx == selected_indices[0] else "")
            ax_hu.plot(x_ref, hu_pred[idx, :], 
                    color=color, linestyle='--', label='Predicted' if idx == selected_indices[0] else "")
            
        # Add labels and titles
        ax_s.set_title('Surface Height (s)')
        ax_s.set_xlabel('x [m]')
        ax_s.set_ylabel('s [m]')
        ax_s.grid(True)

        ax_u.set_title('Velocity (u)')
        ax_u.set_xlabel('x [m]')
        ax_u.set_ylabel('u [m/s]')
        ax_u.grid(True)

        ax_hu.set_title('Momentum (hu)')
        ax_hu.set_xlabel('x [m]')
        ax_hu.set_ylabel('hu [m/s]')
        ax_hu.grid(True)

        # Create colorbar for time values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar = fig.colorbar(sm, ax=[ax_s, ax_u], orientation='vertical', pad=0.02)
        cbar.set_label('Time [s]')

        # Add legend
        handles, labels = ax_s.get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(handles))

        # Add global title with selected times
        time_str = ", ".join([f"{t:.2f}" for t in selected_times])
        fig.suptitle(f"1D flow over sill: Solution Profiles at Times [{time_str}] s", fontsize=14, y=1.02)

        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust layout for title

        # Save the plot
        save_path = os.path.join(workdir, "figures", config.wandb.name, f"time_slices_{num_slices}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        print(f"Time slice plot saved at {save_path}")
        wandb.log({f"SWE Time Slices {num_slices}": wandb.Image(save_path)})


    # -------------- Compute wave speeds --------------

    def compute_wave_speeds(u, h, g):
        c_minus = u - np.sqrt(g * h)
        c_plus  = u + np.sqrt(g * h)
        return c_minus, c_plus
    
    c_minus_pred, c_plus_pred = compute_wave_speeds(u_pred, h_pred, config.setup.g)
    c_minus_ref, c_plus_ref = compute_wave_speeds(u_ref, h_ref, config.setup.g)

    # Absolute error
    wave_cp_error = np.mean(np.abs(c_plus_pred - c_plus_ref), axis=1)
    mae_wave_cp = np.mean(wave_cp_error)
    wave_cm_error = np.mean(np.abs(c_minus_pred - c_minus_ref), axis=1)
    mae_wave_cm = np.mean(wave_cm_error)

    logging.info(f"Right wave (c+) error: {wave_cp_error}")
    logging.info(f"Left wave (c-) error: {wave_cm_error}")
    logging.info(f"Right wave (c+) MAE: {mae_wave_cp}")
    logging.info(f"Left wave (c-) MAE: {mae_wave_cm}")

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_ref, wave_cp_error, label="Right-going wave error (c+)", color='blue', linewidth=2)
    plt.plot(t_ref, wave_cm_error, label="Left-going wave error (c−)", color='red', linewidth=2)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Mean Absolute Wave Speed Error", fontsize=12)
    plt.title("Wave speed error over time", fontsize=14)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save png
    file_name = f'swe_hump_wave_speed.png'
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    wandb.log({"SWE wave speed error": wandb.Image(save_path)})

    # -------------- Create gif --------------

    # Determine axis limits
    x_min, x_max = x_ref.min(), x_ref.max()
    s_min, s_max = np.minimum(s_ref, b_ref).min(), np.maximum(s_ref, b_ref).max()
    u_min, u_max = u_ref.min(), u_ref.max()
    hu_min, hu_max = hu_ref.min(), hu_ref.max()

    # Create figure and axes
    fig, (ax_s, ax_u, ax_hu) = plt.subplots(1, 3, figsize=(24, 6))
    fig.subplots_adjust(bottom=0.25)  # Make room for the legend

    # Set titles, labels, and axis limits with larger fonts
    for ax in [ax_s, ax_u, ax_hu]:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    ax_s.set_title('Surface Height (s)', fontsize=16)
    ax_s.set_xlabel('x [m]', fontsize=14)
    ax_s.set_ylabel('s [m]', fontsize=14)
    ax_s.set_xlim(x_min, x_max)
    ax_s.set_ylim(s_min, s_max)

    ax_u.set_title('Velocity (u)', fontsize=16)
    ax_u.set_xlabel('x [m]', fontsize=14)
    ax_u.set_ylabel('u [m/s]', fontsize=14)
    ax_u.set_xlim(x_min, x_max)
    ax_u.set_ylim(u_min, u_max)

    ax_hu.set_title('Momentum (hu)', fontsize=16)
    ax_hu.set_xlabel('x [m]', fontsize=14)
    ax_hu.set_ylabel('hu [m/s]', fontsize=14)
    ax_hu.set_xlim(x_min, x_max)
    ax_hu.set_ylim(hu_min, hu_max)

    # Initialize plot lines
    line_s_ref, = ax_s.plot([], [], color='blue', linestyle='-', label='Reference')
    line_s_pred, = ax_s.plot([], [], color='red', linestyle='--', label='Predicted')
    line_bath, = ax_s.plot([], [], color='black', linestyle=':', label='Bathymetry')

    line_u_ref, = ax_u.plot([], [], color='blue', linestyle='-')
    line_u_pred, = ax_u.plot([], [], color='red', linestyle='--')

    line_hu_ref, = ax_hu.plot([], [], color='blue', linestyle='-')
    line_hu_pred, = ax_hu.plot([], [], color='red', linestyle='--')

    # Initialization function
    def init():
        line_s_ref.set_data([], [])
        line_s_pred.set_data([], [])
        line_bath.set_data([], [])
        line_u_ref.set_data([], [])
        line_u_pred.set_data([], [])
        line_hu_ref.set_data([], [])
        line_hu_pred.set_data([], [])
        fig.suptitle('')
        return (line_s_ref, line_s_pred, line_bath,
                line_u_ref, line_u_pred,
                line_hu_ref, line_hu_pred)

    # Animation update function
    def update(frame):
        line_s_ref.set_data(x_ref, s_ref[frame])
        line_s_pred.set_data(x_ref, s_pred[frame])
        line_bath.set_data(x_ref, b_ref[frame])

        line_u_ref.set_data(x_ref, u_ref[frame])
        line_u_pred.set_data(x_ref, u_pred[frame])

        line_hu_ref.set_data(x_ref, hu_ref[frame])
        line_hu_pred.set_data(x_ref, hu_pred[frame])

        fig.suptitle(f"1D flow over sill: Solution Profiles at Time {t_ref[frame]:.2f} s", fontsize=18)
        return (line_s_ref, line_s_pred, line_bath,
                line_u_ref, line_u_pred,
                line_hu_ref, line_hu_pred)

    # Add legend at the bottom
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=14)
    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=len(handles), fontsize=14)

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(t_ref), init_func=init, blit=True)

    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save as GIF
    file_name = "swe_fields.gif"
    save_path = os.path.join(save_dir, file_name)
    ani.save(save_path, writer=animation.PillowWriter(fps=10))
    print(f"Animation saved at {save_path}")

    wandb.log({f"SWE u,h,hu fields": wandb.Video(os.path.join(save_dir, file_name))})