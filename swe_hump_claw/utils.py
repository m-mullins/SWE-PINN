import jax.numpy as jnp
import ml_collections
import matplotlib.pyplot as plt
import os
import jax
import numpy as np
import wandb
import optax
import optimistix as optx
from collections.abc import Callable
from typing import Any
from jax.tree_util import tree_flatten


# For cuteflow data
def get_dataset(dataset):
    # Get ref solution
    data = jnp.load(os.path.join("data", dataset), allow_pickle=True)
    x_ref = jnp.array(data['x'])
    # y_ref = jnp.array(data['y'])
    t_ref = jnp.array(data['t'])
    h_ref = jnp.array(data['h'])
    s_ref = jnp.array(data['s'])
    b_ref = jnp.array(data['b'])
    u_ref = jnp.array(data['u'])
    hu_ref = h_ref * u_ref
    # v_ref = jnp.array(data['v'])
    g = data['g']
    manning = 0

    ratio = 1
    x_ref = x_ref[::ratio]
    t_ref = t_ref[::ratio]
    h_ref = h_ref[::ratio, ::ratio]
    s_ref = s_ref[::ratio, ::ratio]
    b_ref = b_ref[::ratio, ::ratio]
    u_ref = u_ref[::ratio, ::ratio]

    # stop = len(t_ref)//2 + 1
    # h_ref = h_ref[:stop]
    # u_ref = u_ref[:stop]
    # b_ref = b_ref[:stop]
    # t_ref = t_ref[:stop]
    # s_ref = s_ref[:stop]
    return (h_ref, u_ref, hu_ref, b_ref, t_ref, x_ref, s_ref, g, manning)


def convert_config_to_dict(config):
    """Converts a ConfigDict object to a plain Python dictionary."""
    if isinstance(config, ml_collections.ConfigDict):
        return {k: convert_config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_config_to_dict(v) for v in config]
    else:
        return config


def get_bc_coords_(dom, t_star, x_star, y_star):
    t0, t1 = dom[0]
    x0, x1 = dom[1]
    y0, y1 = dom[2]
    nt = t_star.shape[0]

    # Slip BC coordinates (bot and top walls)   
    bottom_wall = jnp.stack([x_star, jnp.full_like(x_star, y0)], axis=1)    # y=y0 (bottom)
    top_wall    = jnp.stack([x_star, jnp.full_like(x_star, y1)], axis=1)    # y=y1 (top)
    slip_coords = {
        "bot": bottom_wall,
        "top": top_wall,
    }

    # Outflow BC coordinates (right wall)
    right_wall  = jnp.stack([jnp.full_like(y_star, x1), y_star], axis=1)    # x=x1 (right)
    outflow_coords = jnp.vstack([right_wall])

    # Inflow BC coordinates (left wall)
    left_wall   = jnp.stack([jnp.full_like(y_star, x0), y_star], axis=1)    # x=x0 (left)
    inflow_coords = jnp.vstack([left_wall])

    # Replicate across time steps and combine
    bc_coords = {}
    for wall in ["bot", "top"]:
        coords_tiled = jnp.tile(slip_coords[wall], (nt, 1))
        time_coords = jnp.repeat(t_star, slip_coords[wall].shape[0])[:, None]
        bc_coords[wall] = jnp.hstack([time_coords, coords_tiled])
    
    outflow_coords_tiled = jnp.tile(outflow_coords, (nt, 1))
    time_outflow = jnp.repeat(t_star, outflow_coords.shape[0])[:, None]
    bc_coords["outflow"] = jnp.hstack([time_outflow, outflow_coords_tiled])

    inflow_coords_tiled = jnp.tile(inflow_coords, (nt, 1))
    time_inflow = jnp.repeat(t_star, inflow_coords.shape[0])[:, None]
    bc_coords["inflow"] = jnp.hstack([time_inflow, inflow_coords_tiled])

    return bc_coords

def get_bc_coords(dom, t_star):
    t0, t1 = dom[0]
    x0, x1 = dom[1]
    nt = t_star.shape[0]

    # Solid wall coordinates (left and right walls)
    left_wall = jnp.array([[x0]])  # x=x0 (left)
    right_wall = jnp.array([[x1]])  # x=x1 (right)

    # Replicate across time steps and combine
    bc_coords = {}
    
    # Process left wall
    left_coords_tiled = jnp.tile(left_wall, (nt, 1))
    time_left = jnp.repeat(t_star, left_wall.shape[0])[:, None]
    bc_coords["left_wall"] = jnp.hstack([time_left, left_coords_tiled])

    # Process right wall
    right_coords_tiled = jnp.tile(right_wall, (nt, 1))
    time_right = jnp.repeat(t_star, right_wall.shape[0])[:, None]
    bc_coords["right_wall"] = jnp.hstack([time_right, right_coords_tiled])

    return bc_coords


def g_schedule_step(step, g_min, g_max, train_steps, n=5):
    """
    Computes the gravity constant g using a step function schedule.

    Parameters:
        step: The current training step (JAX array or scalar).
        g_min (float): The initial gravity constant.
        g_max (float): The final gravity constant.
        train_steps (int): The total number of training steps.
        n (int): The number of discrete g values.

    Returns:
        float: The gravity constant at the given step.
    """
    # Ensure n is at least 1 to avoid division by zero
    if n < 1:
        raise ValueError("The number of discrete values n must be at least 1.")
    
    # Calculate the interval of steps for each g value
    interval = train_steps / n
    
    # Determine the current index based on the step
    index = jnp.floor(step / interval).astype(jnp.int32)  # Use JAX's floor and astype
    index = jnp.minimum(index, n - 1)  # Use JAX's minimum instead of Python's min
    
    # Calculate the step size for g values
    g_step = (g_max - g_min) / (n - 1) if n > 1 else 0
    
    # Compute the current g value
    g_value = g_min + index * g_step
    
    return g_value


def g_schedule_sigmoid(step, g_min, g_max, train_steps, k=10):
    """
    Computes the gravity constant g using a sigmoid-based schedule.

    Parameters:
        step: The current training step (JAX array or scalar).
        g_min (float): The initial gravity constant.
        g_max (float): The final gravity constant.
        train_steps (int): The total number of training steps.
        k (float): The steepness of the sigmoid curve. Higher values make the transition sharper.

    Returns:
        float: The gravity constant at the given step.
    """
    # Normalize the step to the range [0, 1]
    t = step / train_steps

    # Compute the sigmoid transition
    sigmoid = 1 / (1 + jnp.exp(-k * (t - 0.5)))

    # Scale the sigmoid output to the range [g_min, g_max]
    g_value = g_min + (g_max - g_min) * sigmoid

    return g_value


def plot_g_schedule(config, workdir, schedule_func, g_max):
    """
    Plots the evolution of g through training.
    """
    steps = np.arange(config.training.max_steps + 1)
    if schedule_func == "step":
        g_values = [g_schedule_step(step, config.training.g_min, g_max, config.training.max_steps, n=5) for step in steps]
    elif schedule_func == "sigmoid":
        g_values = [g_schedule_sigmoid(step, config.training.g_min, g_max, config.training.max_steps, k=10) for step in steps]
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, g_values, label=schedule_func)
    plt.xlabel("Training Steps")
    plt.ylabel("Gravity constant g")
    plt.title("Evolution of g during training")
    plt.legend()
    plt.grid()

    # Show the plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'g_evolution.png'), dpi=300)

    wandb.log({f"g evolution": wandb.Image(os.path.join(save_dir, 'g_evolution.png'))})


def count_params(state):
    """
    Calculate the total number of parameters in the model.

    Args:
        state: The TrainState object containing the model state.

    Returns:
        int: The total number of parameters.
    """
    # Extract the parameters from the state
    params = state.params

    # Flatten the parameter tree and sum the sizes of all arrays
    total_params = sum(jnp.prod(jnp.array(p.shape)) for p in tree_flatten(params)[0])

    total_bytes = 0
    
    # Flatten the parameter tree and iterate over leaf arrays
    for param in tree_flatten(params)[0]:
        total_bytes += param.size * param.dtype.itemsize

    # Convert bytes to MB
    total_mb = total_bytes / (1024 ** 2)
    return total_params, total_mb


class BFGSTrustRegion(optx.AbstractBFGS):
    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = False
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()


def plot_rba_weights(config, workdir, rba_weights, batch, step):
    """
    Plot RBA weights as discrete points at collocation coordinates.
    
    Args:
        rba_weights: JAX array of shape [num_devices, batch_size] (pmap) 
                     or [batch_size] (single device)
        batch: Collocation points array of shape [batch_size, 2] (t, x)
        step: Current training step for title
    """
    # Handle pmap replication
    if rba_weights.ndim > 1:
        rba_weights = rba_weights[0]  # Take first device's weights
    if batch.ndim > 2:
        batch = batch[0]  # Take first device's batch

    # Extract coordinates
    t_coords = batch[:, 0]
    x_coords = batch[:, 1]

    # Create plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_coords,
        t_coords,
        c=rba_weights,
        cmap='viridis',
        s=5,  # Marker size
        edgecolor=None, # 'k'
        linewidth=4.5,
        marker='o'
    )

    # Add text annotations for each point
    # for i, (x, t) in enumerate(zip(x_coords, t_coords)):
    #     plt.text(x + 0.01, t, str(i), fontsize=6, color='black', ha='left', va='center')

    # Determine grid lines
    # n = sqrt(batch_size) + 1. For batch size of 256, n = 16 + 1 = 17.
    batch_size = batch.shape[0]
    n = int(np.sqrt(batch_size)) + 1

    # Compute limits for x and t coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    t_min, t_max = t_coords.min(), t_coords.max()
    # x_min, x_max = -0.998, 0.998
    # t_min, t_max = 0.0, 1.01

    # Compute equally spaced positions
    x_positions = np.linspace(x_min, x_max, n)
    t_positions = np.linspace(t_min, t_max, n)

    # # Add vertical grid lines
    # for x_val in x_positions:
    #     plt.axvline(x=x_val, color='red', linestyle='--', linewidth=0.5)
    # # Add horizontal grid lines
    # for t_val in t_positions:
    #     plt.axhline(y=t_val, color='red', linestyle='--', linewidth=0.5)

    
    plt.colorbar(scatter, label='RBA Weight')
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.title(f'RBA Weights at Training Step {step}')
    
    # Set aspect ratio based on data ranges
    x_range = x_coords.max() - x_coords.min()
    t_range = t_coords.max() - t_coords.min()
    plt.gca().set_aspect(x_range / t_range)

    # Save and log
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'rba_{step}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    wandb.log({f"rba weights step {step}": wandb.Image(os.path.join(save_dir, f'rba_{step}.png'))})


def plot_colloc_pts(config, workdir, batch, step, dom):
    """
    Plot collocation points for a given step.
    
    Args:
        batch: Collocation points array of shape [batch_size, 2] (t, x)
        step: Current training step for title
    """
    # Handle pmap replication
    if batch.ndim > 2:
        batch = batch[0]  # Take first device's batch

    # Extract coordinates
    t_coords = batch[:, 0]
    x_coords = batch[:, 1]

    # Create plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_coords,
        t_coords,
        s=10,  # Marker size
        edgecolor=None, # 'k'
        linewidth=0.5,
        marker='o'
    )

    # # Add text annotations for each point
    # for i, (x, t) in enumerate(zip(x_coords, t_coords)):
    #     plt.text(x + 0.01, t, str(i), fontsize=6, color='black', ha='left', va='center')
    
    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.title(f'Collocation points at Training Step {step} (k={config.weighting.rad_k}, c={config.weighting.rad_c})')
    
    # Set aspect ratio based on data ranges
    plt.xlim(dom[1, 0], dom[1, 1])  # Set x-axis (spatial coordinate) limits
    plt.ylim(dom[0, 0], dom[0, 1])  # Set y-axis (time coordinate) limits
    # x_range = 2 # dom[1,1] - dom[1,0] # x_coords.max() - x_coords.min()
    # t_range = dom[0,1] - dom[0,0] # t_coords.max() - t_coords.min()
    # plt.gca().set_aspect(x_range / t_range)

    # Save and log
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'colloc_{step}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    wandb.log({f"collocation points step {step}": wandb.Image(os.path.join(save_dir, f'colloc_{step}.png'))})


def plot_residuals(config, workdir, residuals, s0, step, dom):
    """
    Plot the residuals as a color map.

    Args:
        residuals: Array of residual values corresponding to each point in s0 (shape: [num_points,])
        s0: Array of point coordinates of shape [num_points, 2] (t, x)
        step: Current training step (for title/annotation)
        dom: Domain array of shape [2, 2]: [[t_min, t_max], [x_min, x_max]]
    """
    # If s0 has extra device dimensions (from pmap), take the first device's data.
    if s0.ndim > 2:
        s0 = s0[0]
        residuals = residuals[0]

    # Extract coordinates
    t_coords = s0[:, 0]
    x_coords = s0[:, 1]

    # Create the plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_coords,
        t_coords,
        c=residuals,
        cmap='viridis',
        s=10,  # Marker size
        edgecolor='none'
    )
    plt.colorbar(scatter, label='Residual value')

    plt.xlabel('Spatial Coordinate (x)')
    plt.ylabel('Time (t)')
    plt.title(f'Residuals at Training Step {step} (k={config.weighting.rad_k}, c={config.weighting.rad_c})')

    # Set the axis limits based on domain. Here, dom[0] is for time and dom[1] for space.
    plt.xlim(dom[1, 0], dom[1, 1])
    plt.ylim(dom[0, 0], dom[0, 1])

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'residuals_{step}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Log the figure to wandb.
    wandb.log({f"residuals step {step}": wandb.Image(filename)})