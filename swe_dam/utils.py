import jax.numpy as jnp
import ml_collections
import matplotlib.pyplot as plt
import os
import jax
import numpy as np
import wandb
from jax.tree_util import tree_flatten
from jax import grad, lax

# def get_dataset():
#     data = jnp.load("data/ns_unsteady_coarse.npy", allow_pickle=True).item()
#     u_ref = jnp.array(data["u"])
#     v_ref = jnp.array(data["v"])
#     p_ref = jnp.array(data["p"])
#     t = jnp.array(data["t"])
#     coords = jnp.array(data["coords"])
#     inflow_coords = jnp.array(data["inflow_coords"])
#     outflow_coords = jnp.array(data["outflow_coords"])
#     wall_coords = jnp.array(data["wall_coords"])
#     cylinder_coords = jnp.array(data["cylinder_coords"])
#     nu = jnp.array(data["nu"])

#     return (
#         u_ref,
#         v_ref,
#         p_ref,
#         coords,
#         inflow_coords,
#         outflow_coords,
#         wall_coords,
#         cylinder_coords,
#         nu,
#     )


# def get_domain():
#     data = jnp.load("data/rect_dom.npy", allow_pickle=True).item()
#     t = jnp.array(data["t"])
#     coords = jnp.array(data["coords"])
#     inflow_coords = jnp.array(data["inflow_coords"])
#     outflow_coords = jnp.array(data["outflow_coords"])
#     wall_coords = jnp.array(data["wall_coords"])
#     nu = jnp.array(data["nu"])
#     u0 = jnp.array(data["u0"])
#     v0 = jnp.array(data["v0"])
#     h0 = jnp.array(data["h0"])
#     x = jnp.array(data["x"])
#     y = jnp.array(data["y"])
#     g = jnp.array(data["g"])

#     return (
#         u0,
#         v0,
#         h0,
#         t,
#         coords,
#         inflow_coords,
#         outflow_coords,
#         wall_coords,
#         nu,
#         x,
#         y,
#         g,
#     )

# For pyclaw data
# def get_dataset(dataset):
#     # Get ref solution
#     data = jnp.load(os.path.join("data", dataset), allow_pickle=True)
#     t_ref = jnp.array(data['times'])
#     h_ref = jnp.array(data['h'])
#     u_ref = jnp.array(data['u'])
#     v_ref = jnp.array(data['v'])
#     b_ref = jnp.array(data['b'])
#     b_ref = jnp.array(b_ref[jnp.newaxis,:,:])
#     b_ref = jnp.repeat(b_ref,h_ref.shape[0],axis=0)
#     x_ref = jnp.array(data['x_coords'])
#     y_ref = jnp.array(data['y_coords'])
#     s_ref = h_ref + b_ref
#     # s_ref = np.array(s_ref)
#     g = data['g']

#     # t_ref = t_ref[:-1]
#     # h_ref = h_ref[1:]
#     # u_ref = u_ref[1:]
#     # v_ref = v_ref[1:]
#     # b_ref = b_ref[1:]
#     # s_ref = s_ref[1:]

#     return (
#         h_ref, u_ref, v_ref, b_ref, t_ref, x_ref, y_ref, s_ref, g
#     )

# For cuteflow data
def get_dataset(dataset):
    # Get ref solution
    data = jnp.load(os.path.join("data", dataset), allow_pickle=True)
    x_ref = jnp.array(data['x'])
    y_ref = jnp.array(data['y'])
    t_ref = jnp.array(data['t'])
    h_ref = jnp.array(data['h'])
    s_ref = jnp.array(data['s'])
    b_ref = jnp.array(data['b'])
    u_ref = jnp.array(data['u'])
    v_ref = jnp.array(data['v'])
    g = data['g']
    manning = 0

    return (
        h_ref, u_ref, v_ref, b_ref, t_ref, x_ref, y_ref, s_ref, g, manning
    )

# def get_fine_mesh():
#     data = jnp.load("data/fine_mesh_coarse.npy", allow_pickle=True).item()
#     fine_coords = jnp.array(data["coords"])

#     data = jnp.load("data/fine_mesh_near_cylinder_coarse.npy", allow_pickle=True).item()
#     fine_coords_near_cyl = jnp.array(data["coords"])

#     return fine_coords, fine_coords_near_cyl

def convert_config_to_dict(config):
    """Converts a ConfigDict object to a plain Python dictionary."""
    if isinstance(config, ml_collections.ConfigDict):
        return {k: convert_config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_config_to_dict(v) for v in config]
    else:
        return config
    
def plot_coordinates(config, workdir):
    # Get dataset
    (   u_ref,
        v_ref,
        h_ref,
        t,
        coords,
        inflow_coords,
        outflow_coords,
        wall_coords,
        nu,
        x,
        y,
        g,
    ) = get_dataset()

    # Extract x and y coordinates
    coords_xy = coords[:, :2]
    inflow_xy = inflow_coords[:, :2]
    outflow_xy = outflow_coords[:, :2]
    wall_xy = wall_coords[:, :2]

    # Create the plot
    plt.figure(figsize=(16, 4))

    # Plot each set of coordinates with unique markers and colors
    plt.scatter(coords_xy[:, 0], coords_xy[:, 1], label="Domain Interior", color="blue", s=10, alpha=0.6)
    plt.scatter(inflow_xy[:, 0], inflow_xy[:, 1], label="Inflow Boundary", color="green", marker=">", s=40, alpha=0.8)
    plt.scatter(outflow_xy[:, 0], outflow_xy[:, 1], label="Outflow Boundary", color="red", marker="<", s=40, alpha=0.8)
    plt.scatter(wall_xy[:, 0], wall_xy[:, 1], label="Wall Boundary", color="purple", marker="s", s=30, alpha=0.8)

    # Add labels and legend
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.title("Coordinate Visualization", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.axis("equal")  # Ensure equal aspect ratio

    # Show the plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'collocation_points.png'), dpi=300)

def get_bc_coords(dom, t_star, x_star, y_star):
    t0, t1 = dom[0]
    x0, x1 = dom[1]
    y0, y1 = dom[2]
    nt = t_star.shape[0]

    # Slip BC coordinates (right and top walls)
    right_wall = jnp.stack([jnp.full_like(y_star, x1), y_star], axis=1)  # x=x1 (right)
    top_wall = jnp.stack([x_star, jnp.full_like(x_star, y1)], axis=1)     # y=y1 (top)
    slip_coords = {
        "right": right_wall,
        "top": top_wall,
    }

    # Outflow BC coordinates (left and bottom walls)
    left_wall = jnp.stack([jnp.full_like(y_star, x0), y_star], axis=1)   # x=x0 (left)
    bottom_wall = jnp.stack([x_star, jnp.full_like(x_star, y0)], axis=1)  # y=y0 (bottom)
    outflow_coords = jnp.vstack([left_wall, bottom_wall])

    # Replicate across time steps and combine
    bc_coords = {}
    for wall in ["right", "top"]:
        coords_tiled = jnp.tile(slip_coords[wall], (nt, 1))
        time_coords = jnp.repeat(t_star, slip_coords[wall].shape[0])[:, None]
        bc_coords[wall] = jnp.hstack([time_coords, coords_tiled])
    
    outflow_coords_tiled = jnp.tile(outflow_coords, (nt, 1))
    time_outflow = jnp.repeat(t_star, outflow_coords.shape[0])[:, None]
    bc_coords["outflow"] = jnp.hstack([time_outflow, outflow_coords_tiled])

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


def clipped_second_derivative(f, argnum, clip_val):
    """Returns a function that computes the clipped second derivative."""
    def first_grad(*args):
        g = grad(f, argnums=argnum)(*args)
        return lax.clamp(-clip_val, g, clip_val)
    
    def second_grad(*args):
        return grad(first_grad, argnums=argnum)(*args)
    
    return second_grad