import os
import ml_collections
import wandb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jaxpi.utils import restore_checkpoint
from absl import logging

import models
from utils import get_dataset, get_bc_coords_values


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, NU = get_dataset(os.path.join("data", config.dataset))
    # u_ref = u_ref[::12,::12,::12]
    # v_ref = v_ref[::12,::12,::12]
    # t_star = t_star[::12]
    # x_star = x_star[::12]
    # y_star = y_star[::12]
    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]], [y_star[0], y_star[-1]]])
    bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.Burgers2d(config, u0, v0, t_star, x_star, y_star, bc_coords, bc_values)
    ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_u_error, l2_v_error = model.compute_l2_error(params, t_star, x_star, y_star, u_ref, v_ref)
    print("L2 u error: {:.3e}".format(l2_u_error))
    print("L2 v error: {:.3e}".format(l2_v_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
    v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)

    Nt, Nx, Ny = u_ref.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set up the figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # First row (u: Ground truth, Prediction, Absolute Error)
    pcm1 = axes[0, 0].pcolormesh(X, Y, u_ref[0], cmap='jet', shading='gouraud')
    pcm2 = axes[0, 1].pcolormesh(X, Y, u_pred[0], cmap='jet', shading='gouraud')
    pcm3 = axes[0, 2].pcolormesh(X, Y, jnp.abs(u_ref[0] - u_pred[0]), cmap='jet', shading='gouraud')
    fig.colorbar(pcm1, ax=axes[0, 0])
    fig.colorbar(pcm2, ax=axes[0, 1])
    fig.colorbar(pcm3, ax=axes[0, 2])

    # Titles for u
    axes[0, 0].set_title('Ground Truth u')
    axes[0, 1].set_title('Predicted u')
    axes[0, 2].set_title('Absolute Error u')

    # Second row (v: Ground truth, Prediction, Absolute Error)
    pcm4 = axes[1, 0].pcolormesh(X, Y, v_ref[0], cmap='jet', shading='gouraud')
    pcm5 = axes[1, 1].pcolormesh(X, Y, v_pred[0], cmap='jet', shading='gouraud')
    pcm6 = axes[1, 2].pcolormesh(X, Y, jnp.abs(v_ref[0] - v_pred[0]), cmap='jet', shading='gouraud')
    fig.colorbar(pcm4, ax=axes[1, 0])
    fig.colorbar(pcm5, ax=axes[1, 1])
    fig.colorbar(pcm6, ax=axes[1, 2])

    # Titles for v
    axes[1, 0].set_title('Ground Truth v')
    axes[1, 1].set_title('Predicted v')
    axes[1, 2].set_title('Absolute Error v')

    # Third row (Quiver: Ground truth, Prediction)
    quiver_exact = axes[2, 0].quiver(X, Y, u_ref[0], v_ref[0], scale=10, color='black')
    quiver_pred = axes[2, 1].quiver(X, Y, u_pred[0], v_pred[0], scale=10, color='black')
    fig.colorbar(quiver_exact, ax=axes[2, 0])
    fig.colorbar(quiver_pred, ax=axes[2, 1])

    axes[2, 0].set_title('Exact Velocity Field')
    axes[2, 1].set_title('Predicted Velocity Field')

    # Function to update the plots for animation
    def update(frame):
        # Update pcolormesh for u
        pcm1.set_array(u_ref[frame].ravel())
        pcm2.set_array(u_pred[frame].ravel())
        pcm3.set_array(jnp.abs(u_ref[frame] - u_pred[frame]).ravel())

        # Update pcolormesh for v
        pcm4.set_array(v_ref[frame].ravel())
        pcm5.set_array(v_pred[frame].ravel())
        pcm6.set_array(jnp.abs(v_ref[frame] - v_pred[frame]).ravel())

        # Update quiver plots for velocity field
        quiver_exact.set_UVC(u_ref[frame], v_ref[frame])
        quiver_pred.set_UVC(u_pred[frame], v_pred[frame])

        # Update titles
        time = dt * frame
        axes[0, 0].set_title(f'Ground Truth u at time {time:.2f} s')
        axes[0, 1].set_title(f'Predicted u at time {time:.2f} s')
        axes[0, 2].set_title(f'Absolute Error u at time {time:.2f} s')
        axes[1, 0].set_title(f'Ground Truth v at time {time:.2f} s')
        axes[1, 1].set_title(f'Predicted v at time {time:.2f} s')
        axes[1, 2].set_title(f'Absolute Error v at time {time:.2f} s')
        axes[2, 0].set_title(f'Exact Velocity Field at time {time:.2f} s')
        axes[2, 1].set_title(f'Predicted Velocity Field at time {time:.2f} s')

        return pcm1, pcm2, pcm3, pcm4, pcm5, pcm6, quiver_exact, quiver_pred

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, blit=True)

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'burgers2d_Re_{config.reynolds}.gif'
    ani.save(os.path.join(save_dir, file_name), writer=animation.PillowWriter(fps=10))

    # Clear the figure to prevent overlap of subsequent samples
    plt.clf()

    wandb.log({f"Burgers2d Re{config.reynolds} uv fields": wandb.Video(os.path.join(save_dir, file_name))})



def evaluate_s2s(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, NU = get_dataset(os.path.join("data", config.dataset))
    u_ref = u_ref[:-1, :]
    v_ref = v_ref[:-1, :]
    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]], [y_star[0], y_star[-1]]])
    bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.Burgers2d(config, u0, v0, t, x_star, y_star, bc_coords, bc_values)

    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        u = u_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_ref[num_time_steps * idx : num_time_steps * (idx + 1), :]

        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_u_error, l2_v_error = model.compute_l2_error(params, t, x_star, y_star, u, v)
        logging.info("Time window: {}, u error: {:.3e}, v error: {:.3e}".format(idx + 1, l2_u_error, l2_v_error))

        u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)

        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
    v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)

    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))

    Nt, Nx, Ny = u_ref.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set up the figure and axes
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # First row (u: Ground truth, Prediction, Absolute Error)
    pcm1 = axes[0, 0].pcolormesh(X, Y, u_ref[0], cmap='jet', shading='gouraud')
    pcm2 = axes[0, 1].pcolormesh(X, Y, u_pred[0], cmap='jet', shading='gouraud')
    pcm3 = axes[0, 2].pcolormesh(X, Y, jnp.abs(u_ref[0] - u_pred[0]), cmap='jet', shading='gouraud')
    fig.colorbar(pcm1, ax=axes[0, 0])
    fig.colorbar(pcm2, ax=axes[0, 1])
    fig.colorbar(pcm3, ax=axes[0, 2])

    # Titles for u
    axes[0, 0].set_title('Ground Truth u')
    axes[0, 1].set_title('Predicted u')
    axes[0, 2].set_title('Absolute Error u')

    # Second row (v: Ground truth, Prediction, Absolute Error)
    pcm4 = axes[1, 0].pcolormesh(X, Y, v_ref[0], cmap='jet', shading='gouraud')
    pcm5 = axes[1, 1].pcolormesh(X, Y, v_pred[0], cmap='jet', shading='gouraud')
    pcm6 = axes[1, 2].pcolormesh(X, Y, jnp.abs(v_ref[0] - v_pred[0]), cmap='jet', shading='gouraud')
    fig.colorbar(pcm4, ax=axes[1, 0])
    fig.colorbar(pcm5, ax=axes[1, 1])
    fig.colorbar(pcm6, ax=axes[1, 2])

    # Titles for v
    axes[1, 0].set_title('Ground Truth v')
    axes[1, 1].set_title('Predicted v')
    axes[1, 2].set_title('Absolute Error v')

    # Third row (Quiver: Ground truth, Prediction)
    quiver_exact = axes[2, 0].quiver(X, Y, u_ref[0], v_ref[0], scale=10, color='black')
    quiver_pred = axes[2, 1].quiver(X, Y, u_pred[0], v_pred[0], scale=10, color='black')
    fig.colorbar(quiver_exact, ax=axes[2, 0])
    fig.colorbar(quiver_pred, ax=axes[2, 1])

    axes[2, 0].set_title('Exact Velocity Field')
    axes[2, 1].set_title('Predicted Velocity Field')

    # Function to update the plots for animation
    def update(frame):
        # Update pcolormesh for u
        pcm1.set_array(u_ref[frame].ravel())
        pcm2.set_array(u_pred[frame].ravel())
        pcm3.set_array(jnp.abs(u_ref[frame] - u_pred[frame]).ravel())

        # Update pcolormesh for v
        pcm4.set_array(v_ref[frame].ravel())
        pcm5.set_array(v_pred[frame].ravel())
        pcm6.set_array(jnp.abs(v_ref[frame] - v_pred[frame]).ravel())

        # Update quiver plots for velocity field
        quiver_exact.set_UVC(u_ref[frame], v_ref[frame])
        quiver_pred.set_UVC(u_pred[frame], v_pred[frame])

        # Update titles
        time = dt * frame
        axes[0, 0].set_title(f'Ground Truth u at time {time:.2f} s')
        axes[0, 1].set_title(f'Predicted u at time {time:.2f} s')
        axes[0, 2].set_title(f'Absolute Error u at time {time:.2f} s')
        axes[1, 0].set_title(f'Ground Truth v at time {time:.2f} s')
        axes[1, 1].set_title(f'Predicted v at time {time:.2f} s')
        axes[1, 2].set_title(f'Absolute Error v at time {time:.2f} s')
        axes[2, 0].set_title(f'Exact Velocity Field at time {time:.2f} s')
        axes[2, 1].set_title(f'Predicted Velocity Field at time {time:.2f} s')

        return pcm1, pcm2, pcm3, pcm4, pcm5, pcm6, quiver_exact, quiver_pred

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, blit=True)

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'burgers2d_Re_{config.reynolds}.gif'
    ani.save(os.path.join(save_dir, file_name), writer=animation.PillowWriter(fps=10))

    # Clear the figure to prevent overlap of subsequent samples
    plt.clf()

    wandb.log({f"Burgers2d Re{config.reynolds} uv fields": wandb.Video(os.path.join(save_dir, file_name))})


def plot_sliced_results(config: ml_collections.ConfigDict, workdir: str):
    u_ref, v_ref, t_star, x_star, y_star, NU = get_dataset(os.path.join("data", config.dataset))
    u0 = u_ref[0, :]
    v0 = v_ref[0, :]
    dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]], [y_star[0], y_star[-1]]])
    bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.Burgers2d(config, u0, v0, t_star, x_star, y_star, bc_coords, bc_values)
    ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt")
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_u_error, l2_v_error = model.compute_l2_error(params, t_star, x_star, y_star, u_ref, v_ref)
    print("L2 u error: {:.3e}".format(l2_u_error))
    print("L2 v error: {:.3e}".format(l2_v_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star, model.y_star)
    v_pred = model.v_pred_fn(params, model.t_star, model.x_star, model.y_star)

    Nt, Nx, Ny = u_ref.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set time steps to plot
    ts = [0,5,10]

    # Compute global max error for uniform colorbar scaling
    max_abs_error = jnp.max(jnp.abs(u_ref - u_pred))
    min_u = min(jnp.min(u_pred),jnp.min(u_ref))
    max_u = max(jnp.max(u_pred),jnp.max(u_ref))

    # Set up the figure and axes
    fig, axes = plt.subplots(len(ts), 3, figsize=(18, 15))

    for idx in range(len(ts)):
        # Row (u: Ground truth, Prediction, Absolute Error)
        pcm1 = axes[idx, 0].pcolormesh(X, Y, u_ref[ts[idx]], cmap='jet', shading='gouraud', vmin=min_u, vmax=max_u)
        pcm2 = axes[idx, 1].pcolormesh(X, Y, u_pred[ts[idx]], cmap='jet', shading='gouraud', vmin=min_u, vmax=max_u)
        pcm3 = axes[idx, 2].pcolormesh(X, Y, jnp.abs(u_ref[ts[idx]] - u_pred[ts[idx]]), cmap='jet', shading='gouraud', vmin=0, vmax=max_abs_error)
        fig.colorbar(pcm1, ax=axes[idx, 0])
        fig.colorbar(pcm2, ax=axes[idx, 1])
        fig.colorbar(pcm3, ax=axes[idx, 2])

        # Titles for u
        axes[idx, 0].set_title(f'Reference u t = {ts[idx]*dt} s')
        axes[idx, 1].set_title(f'Predicted u t = {ts[idx]*dt} s')
        axes[idx, 2].set_title(f'Absolute Error u t = {ts[idx]*dt} s')

        for idx2 in range(3):
            axes[idx, idx2].set_xlabel('x')
            axes[idx, idx2].set_ylabel('y')

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'burgers2d_u_ts.png'
    plt.savefig(os.path.join(save_dir, file_name))

    # Compute global max error for uniform colorbar scaling
    max_abs_error = jnp.max(jnp.abs(v_ref - v_pred))
    min_v = min(jnp.min(v_pred),jnp.min(v_ref))
    max_v = max(jnp.max(v_pred),jnp.max(v_ref))

    # Set up the figure and axes
    fig, axes = plt.subplots(len(ts), 3, figsize=(18, 15))

    for idx in range(len(ts)):
        # Second row (v: Ground truth, Prediction, Absolute Error)
        pcm4 = axes[idx, 0].pcolormesh(X, Y, v_ref[ts[idx]], cmap='jet', shading='gouraud', vmin=min_v, vmax=max_v)
        pcm5 = axes[idx, 1].pcolormesh(X, Y, v_pred[ts[idx]], cmap='jet', shading='gouraud', vmin=min_v, vmax=max_v)
        pcm6 = axes[idx, 2].pcolormesh(X, Y, jnp.abs(v_ref[ts[idx]] - v_pred[ts[idx]]), cmap='jet', shading='gouraud', vmin=0, vmax=max_abs_error)
        fig.colorbar(pcm4, ax=axes[idx, 0])
        fig.colorbar(pcm5, ax=axes[idx, 1])
        fig.colorbar(pcm6, ax=axes[idx, 2])

        # Titles for v
        axes[idx, 0].set_title(f'Reference v t = {ts[idx]*dt} s')
        axes[idx, 1].set_title(f'Predicted v t = {ts[idx]*dt} s')
        axes[idx, 2].set_title(f'Absolute Error v t = {ts[idx]*dt} s')

        for idx2 in range(3):
            axes[idx, idx2].set_xlabel('x')
            axes[idx, idx2].set_ylabel('y')

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'burgers2d_v_ts.png'
    plt.savefig(os.path.join(save_dir, file_name))


