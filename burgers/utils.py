import scipy.io
import ml_collections
import matplotlib.pyplot as plt
import numpy as np
import os
import wandb
from jax.tree_util import tree_leaves

def get_dataset(file_path):
    data = scipy.io.loadmat(file_path)
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    # Downsample ds
    t_star = t_star[::2]
    x_star = x_star[::4]
    u_ref = u_ref[::2, ::4]

    return u_ref, t_star, x_star

def convert_config_to_dict(config):
    """Converts a ConfigDict object to a plain Python dictionary."""
    if isinstance(config, ml_collections.ConfigDict):
        return {k: convert_config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_config_to_dict(v) for v in config]
    else:
        return config

# def plot_collocation_points(batch, file_path):
#     """Plots the collocation points for a given batch."""
#     # Extract x and t coordinates
#     t = batch[:, 0]
#     x = batch[:, 1]

#     # Plot the collocation points
#     plt.figure(figsize=(8, 6))
#     plt.scatter(t, x, c='blue', s=5, alpha=0.6, label='Collocation Points')
#     plt.xlabel('t')
#     plt.ylabel('x')
#     plt.title('Collocation Points')
#     plt.grid(True)
#     # plt.legend()

#     # Save the figure
#     plt.savefig(join(file_path,'collocation_points.png'), dpi=300)
#     plt.close()

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
        t_coords,
        x_coords,
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
    plt.xlabel('Time (t)')
    plt.ylabel('Spatial Coordinate (x)')
    plt.title(f'RBA Weights at Training Step {step}')
    
    # Set aspect ratio based on data ranges
    x_range = x_coords.max() - x_coords.min()
    t_range = t_coords.max() - t_coords.min()
    plt.gca().set_aspect(t_range / x_range)

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
        t_coords,
        x_coords,
        s=10,  # Marker size
        edgecolor=None, # 'k'
        linewidth=0.5,
        marker='o'
    )

    # # Add text annotations for each point
    # for i, (x, t) in enumerate(zip(x_coords, t_coords)):
    #     plt.text(x + 0.01, t, str(i), fontsize=6, color='black', ha='left', va='center')
    
    plt.xlabel('Time (t)')
    plt.ylabel('Spatial Coordinate (x)')
    plt.title(f'Collocation points at Training Step {step} (k={config.weighting.rad_k}, c={config.weighting.rad_c})')
    
    # Set aspect ratio based on data ranges
    plt.xlim(dom[0, 0], dom[0, 1])  # Set x-axis (time coordinate) limits
    plt.ylim(dom[1, 0], dom[1, 1])  # Set y-axis (spatial coordinate) limits
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
        t_coords,
        x_coords,
        c=residuals,
        cmap='viridis',
        s=10,  # Marker size
        edgecolor='none'
    )
    plt.colorbar(scatter, label='Residual value')

    plt.xlabel('Time (t)')
    plt.ylabel('Spatial Coordinate (x)')
    plt.title(f'Residuals at Training Step {step} (k={config.weighting.rad_k}, c={config.weighting.rad_c})')

    # Set the axis limits based on domain. Here, dom[0] is for time and dom[1] for space.
    plt.xlim(dom[0, 0], dom[0, 1])
    plt.ylim(dom[1, 0], dom[1, 1])

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'residuals_{step}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Log the figure to wandb.
    wandb.log({f"residuals step {step}": wandb.Image(filename)})

def get_tree_size_mb(pytree):
    leaves = tree_leaves(pytree)
    size_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in leaves if hasattr(leaf, 'size'))
    size_mb = size_bytes / (1024 * 1024)
    return size_mb