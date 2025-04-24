import CGNS.MAP
import numpy as np
from scipy.interpolate import griddata

file_name = "out_mesh_93031_101" # Radial dam break medium
file_path = f"swe_dam/data/{file_name}.cgns"

# Load the CGNS file
(tree, links, paths) = CGNS.MAP.load(file_path)

# ----------- Raw data -----------

# Extract all FlowSolution nodes (e.g., Solution_0, Solution_1, ...)
solutions = []
for node in tree[2]:  # Iterate through the root node's children
    if node[0] == 'Base':
        base = node
        for zone in base[2]:  # Iterate through the Base's children
            if zone[0] == 'Zone':
                for child in zone[2]:  # Iterate through the Zone's children
                    if child[3] == 'FlowSolution_t':
                        solutions.append(child)

# Initialize storage for variables
h_list, s_list, b_list, u_list, v_list = [], [], [], [], []

for solution in solutions:
    h = s = b = u = v = None
    for var in solution[2]:  # Iterate through the solution's variables
        var_name = var[0]
        if var_name == 'Water Depth':
            h = var[1]
        elif var_name == 'Free surface elevation':
            s = var[1]
        elif var_name == 'Bathymetry':
            b = var[1]
        elif var_name == 'VelocityX':
            u = var[1]
        elif var_name == 'VelocityY':
            v = var[1]
    
    if all([h is not None, s is not None, b is not None, u is not None, v is not None]):
        h_list.append(h)
        s_list.append(s)
        b_list.append(b)
        u_list.append(u)
        v_list.append(v)

# Convert to NumPy arrays
h_array = np.stack(h_list)
s_array = np.stack(s_list)
b_array = np.stack(b_list)
u_array = np.stack(u_list)
v_array = np.stack(v_list)

# Extract grid coordinates
coords = tree[2][1][2][0][2][1][2]
x_coords = np.array(coords[0][1])
y_coords = np.array(coords[1][1])
z_coords = np.array(coords[2][1])

t_coords = np.array(tree[2][1][2][1][2][0][1])


# ----------- Interpolation -----------

xy_res = 101
interpolation_method = "linear" # 'linear', 'nearest', 'cubic'
t_res = len(t_coords)
num_conn = h_array.shape[1]
num_node = len(x_coords)

# Extract connectivity data from the Elements_t node
elements_node = None
for node in tree[2]:
    if node[0] == 'Base':
        base = node
        for zone in base[2]:
            if zone[0] == 'Zone':
                for child in zone[2]:
                    if child[3] == 'Elements_t':
                        elements_node = child
                        break

# Get the connectivity array (indices of nodes for each cell)
connectivity = elements_node[2][1][1]  # ElementConnectivity array
connectivity = connectivity - 1  # Adjust of 0-indexing
connectivity = connectivity.reshape(-1, 3)  # Reshape for triangular cells (9546, 3)

# Compute cell centers (x_centers, y_centers)
x_centers = np.zeros(num_conn)
y_centers = np.zeros(num_conn)
for i in range(num_conn):
    node_indices = connectivity[i]  # Indices of nodes for cell i (e.g., [2393, 2327, 2392])
    x_centers[i] = x_coords[node_indices].mean()
    y_centers[i] = y_coords[node_indices].mean()

# Create a 51x51 grid covering the spatial domain
xi = np.linspace(0, 40, xy_res)
yi = np.linspace(0, 40, xy_res)
# xi = np.linspace(x_centers.min(), x_centers.max(), xy_res)
# yi = np.linspace(y_centers.min(), y_centers.max(), xy_res)
xi_grid, yi_grid = np.meshgrid(xi, yi, indexing='ij')  # Shape (xy_res, xy_res)

# Initialize storage for interpolated data (41 time steps, xy_res x xy_res grid, 5 variables)
interpolated_data = {
    'h': np.zeros((t_res, xy_res, xy_res)),
    's': np.zeros((t_res, xy_res, xy_res)),
    'b': np.zeros((t_res, xy_res, xy_res)),
    'u': np.zeros((t_res, xy_res, xy_res)),
    'v': np.zeros((t_res, xy_res, xy_res))
}

# For each time step
for t in range(len(t_coords)):
    # Extract solutions for time step t
    h_t = h_array[t, :]  # Shape (9546,)
    s_t = s_array[t, :]
    b_t = b_array[t, :]
    u_t = u_array[t, :]
    v_t = v_array[t, :]
    
    # Interpolate with linear + nearest fallback
    for var_name in ['h', 's', 'b', 'u', 'v']:
        data_t = locals()[f"{var_name}_t"]  # Get variable (h_t, s_t, etc.)
        
        # Linear interpolation
        interp_linear = griddata(
            (x_centers, y_centers), data_t, (xi_grid, yi_grid), method='linear'
        )
        # Nearest neighbor fallback
        interp_nearest = griddata(
            (x_centers, y_centers), data_t, (xi_grid, yi_grid), method='nearest'
        )
        # Combine
        interpolated_data[var_name][t] = np.where(
            np.isnan(interp_linear), interp_nearest, interp_linear
        )

output_path_interpolated = f"swe_dam/data/radial_dam_break_data_inter_t{t_res}_xy{xy_res}.npz"

np.savez(
    output_path_interpolated,
    x=xi,
    y=yi,
    t=t_coords,
    h=interpolated_data['h'],
    s=interpolated_data['s'],
    b=interpolated_data['b'],
    u=interpolated_data['u'],
    v=interpolated_data['v'],
    g=9.81
)

print(f"Data saved to {output_path_interpolated}")