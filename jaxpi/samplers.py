from abc import ABC, abstractmethod
from functools import partial

import jax.numpy as jnp
from jax import random, pmap, local_device_count, vmap, debug, tree_util, jit
import jax.profiler
from torch.utils.data import Dataset


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_key=random.PRNGKey(1234)):
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        batch = self.data_generation(keys)
        return batch

    def data_generation(self, key):
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    """
    Generates a random uniform sample of coordinates. 
    Will return a new set of random collocation points each time.
    """
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.uniform(
            key,
            shape=(self.batch_size, self.dim),
            minval=self.dom[:, 0],
            maxval=self.dom[:, 1],
        )

        return batch


class SphereSampler(BaseSampler):

    """
    Generates uniform samples from a 3D unit sphere centered at the origin
    and return phi, theta in spherical coordinates
    """

    def __init__(self, temporal_dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.temporal_dom = temporal_dom

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        batch = random.normal(key, shape=(self.batch_size, 3))
        norm = jnp.linalg.norm(batch, axis=1, keepdims=True)
        batch = batch / norm

        # phi = jnp.arctan2(batch[:, 0:1], xyz_batch[:, 1:2])
        # theta = jnp.arccos(batch[:, 2:3])
        # batch = jnp.concatenate([phi, theta], axis=1)

        if self.temporal_dom is not None:
            key, _ = random.split(key)

            t_batch = random.uniform(
                key,
                shape=(self.batch_size, 1),
                minval=self.temporal_dom[0],
                maxval=self.temporal_dom[1],
            )

            batch = jnp.concatenate([t_batch, batch], axis=1)

        return batch


class SpaceSampler(BaseSampler):
    def __init__(self, coords, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.coords = coords

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        idx = random.choice(key, self.coords.shape[0], shape=(self.batch_size,))
        batch = self.coords[idx, :]

        return batch


class TimeSpaceSampler(BaseSampler):
    def __init__(
        self, temporal_dom, spatial_coords, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)

        self.temporal_dom = temporal_dom        # (2)
        self.spatial_coords = spatial_coords    # (25,3) (n, x/y/b)

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        temporal_batch = random.uniform(
            key1,
            shape=(self.batch_size, 1),
            minval=self.temporal_dom[0],
            maxval=self.temporal_dom[1],
        )   # (32,1)

        spatial_idx = random.choice(
            key2, self.spatial_coords.shape[0], shape=(self.batch_size,)
        )   # (32)
        spatial_batch = self.spatial_coords[spatial_idx, :]                 # (32,3)
        batch = jnp.concatenate([temporal_batch, spatial_batch], axis=1)    # (32,4)

        return batch


class FixedRandomSampler(BaseSampler):
    """
    Generates a random uniform sample of coordinates. 
    Will return the same set of random collocation points each time.
    """
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]
        # Pre-generate fixed collocation points
        self.fixed_batch = random.uniform(
            rng_key,
            shape=(batch_size, self.dim),
            minval=dom[:, 0],
            maxval=dom[:, 1],
        )
        self.idx = 0

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        # Cycle through fixed points in chunks
        batch = self.fixed_batch[self.idx : self.idx + self.batch_size]
        self.idx = (self.idx + self.batch_size) % len(self.fixed_batch)
        return batch


class FixedSampler(BaseSampler):
    """
    Generates a uniform sample of coordinates. 
    Will return the same set of uniformly spaced collocation points each time.
    """
    def __init__(self, dom, batch_size, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom
        self.dim = dom.shape[0]
        
        # Ensure batch_size is a perfect dim-th power for uniform grid
        n_per_dim = int(round(batch_size ** (1 / self.dim)))
        # if n_per_dim ** self.dim != batch_size:
        #     raise ValueError(f"batch_size must be a perfect {self.dim}-th power. Got {batch_size}.")
        
        # Generate evenly spaced points for each dimension
        grids = [jnp.linspace(dom[i, 0], dom[i, 1], n_per_dim) for i in range(self.dim)]
        # Create meshgrid for all dimensions
        mesh = jnp.meshgrid(*grids, indexing='ij')
        # Flatten and stack to form (batch_size, dim) array
        self.fixed_batch = jnp.stack([m.ravel() for m in mesh], axis=1)
        
        self.idx = 0

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        # Cycle through the entire grid in each batch
        batch = self.fixed_batch[self.idx : self.idx + self.batch_size]
        self.idx = (self.idx + self.batch_size) % len(self.fixed_batch)
        return batch
    

class StructuredRandomSampler(BaseSampler):
    """
    Generates structured random samples for RBA training by dividing the spatiotemporal domain
    into N*N groups (where N = sqrt(batch_size)) and sampling one random point per group.

    Note: batch_size must be a perfect square (e.g., 16, 64, 256) to create N*N grid.
    """
    def __init__(self, dom, batch_size, n_groups_x=16, n_groups_t=16, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom  # Shape [2, 2]: [[t_min, t_max], [x_min, x_max]]
        self.batch_size = batch_size

        # Ensure batch_size is a perfect square
        self.n = int(jnp.sqrt(batch_size))
        # if self.n ** 2 != batch_size:
        #     raise ValueError(f"batch_size must be a perfect square. Got {batch_size}")
        
        self.n_groups_x = self.n
        self.n_groups_t = self.n
        
        # Calculate group dimensions
        self.dx = (dom[1][1] - dom[1][0]) / self.n_groups_x
        self.dt = (dom[0][1] - dom[0][0]) / self.n_groups_t
        

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        """Generates one point per group in JIT-friendly way"""
        # Create grid of group indices
        t_indices = jnp.arange(self.n_groups_t)
        x_indices = jnp.arange(self.n_groups_x)
        grid_t, grid_x = jnp.meshgrid(t_indices, x_indices, indexing="ij")
        group_indices = jnp.stack([grid_t.ravel(), grid_x.ravel()], axis=1)

        # Split keys for each group
        subkeys = random.split(key, self.batch_size)

        def sample_group(subkey, indices):
            """Vectorized sampling for a single group"""
            t_idx, x_idx = indices
            
            # Calculate group bounds using corrected domain indices
            x_min = self.dom[1][0] + x_idx * self.dx  # x starts from dom[1][0]
            x_max = x_min + self.dx
            t_min = self.dom[0][0] + t_idx * self.dt  # t starts from dom[0][0]
            t_max = t_min + self.dt
            
            # Sample within group bounds
            return random.uniform(
                subkey,
                shape=(2,),
                minval=jnp.array([t_min, x_min]),
                maxval=jnp.array([t_max, x_max])
            )

        # Vectorize over all groups
        return vmap(sample_group)(subkeys, group_indices)
    

class RADSampler(BaseSampler):
    """
    Residual-based Adaptive Distribution (RAD) sampler.

    This sampler generates a dense candidate set S0 uniformly over the domain,
    computes the residuals using a user-provided residual function, and then samples
    a new set of collocation points according to the PDF:

        p(x) ∝ (ε(x)^k / E[ε(x)^k]) + c

    where:
      - ε(x) is the PDE residual at point x.
      - E[ε(x)^k] is approximated by the mean over the candidate set.
      - k (default=1) controls the sensitivity to the residual.
      - c (default=1) is a stabilization parameter.
    """
    def __init__(self, dom, batch_size, residual_fn=None, k=1.0, c=1.0, rng_key=random.PRNGKey(1234)):
        super().__init__(batch_size, rng_key)
        self.dom = dom                      # Expected shape: [2, 2]: [[t_min, t_max], [x_min, x_max]]
        self.dim = dom.shape[0]             # Here dim is 2 (time and space)
        self.candidate_size = batch_size*4
        if residual_fn is None:
            raise ValueError("A residual function (residual_fn) must be provided.")
        self.residual_fn = residual_fn      # Should be vectorized; input shape (num_points, 2) and output shape (num_points,)
        self.k = k
        self.c = c
        self.key = rng_key

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, keys, state):
        """
        Generates a new set of collocation points by:
          1. Sampling a dense candidate set S0 uniformly over the domain.
          2. Computing the residuals at S0 using the provided residual function.
          3. Forming the PDF:
                p(x) ∝ (ε(x)^k / E[ε(x)^k]) + c,
             where E[ε(x)^k] is approximated by the mean over S0.
          4. Normalizing the PDF into a probability mass function (PMF) and then sampling
             a subset of points from S0 according to the PMF.
        """
        # Sample candidate set S0 uniformly over the domain.
        # Since dom is of shape [2, 2] where row0 is time and row1 is space, we create min and max arrays.
        min_vals = jnp.array([self.dom[0, 0], self.dom[1, 0]])
        max_vals = jnp.array([self.dom[0, 1], self.dom[1, 1]])
        S0 = random.uniform(
            keys,
            shape=(self.candidate_size, self.dim),
            minval=min_vals,
            maxval=max_vals
        )

        # Add a dummy device axis so that S0 becomes shape (1, candidate_size, dim)
        S0_expanded = jnp.expand_dims(S0, axis=0)

        # Compute residuals at candidate points.
        residuals = self.residual_fn(state, S0_expanded) # Expected shape: (1, candidate_size,)
        residuals = jnp.squeeze(residuals, axis=0) # Squeeze out the device axis
        residuals = jnp.abs(residuals)  # Take absolute value of residuals, important to prevent alternate sampling due to power

        # Compute PMF
        res_power = jnp.power(residuals, self.k)    # Compute the numerator: ε(x)^k
        exp_val = jnp.mean(res_power)               # Approximate the expectation E[ε(x)^k] using the mean over the candidate set
        pdf = (res_power / exp_val) + self.c        # Compute the unnormalized PDF
        pmf = pdf / jnp.sum(pdf)                    # Normalize to obtain a probability mass function (PMF)

        # Sample indices from S0 without replacement.
        keys, subkey = random.split(keys)
        selected_indices = random.choice(
            subkey,
            self.candidate_size,
            shape=(self.batch_size,),
            p=pmf,
            replace=False
        )

        # Gather the sampled collocation points.
        batch = S0[selected_indices, :]
        return batch, residuals, S0
    
    def __call__(self, state):
        # Replicate the state along the device axis.
        replicated_state = tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.num_devices,) + x.shape), state)

        # Split our internal key into one key for updating and keys for the devices.
        keys = random.split(self.key, self.num_devices + 1)
        self.key = keys[0]  # update internal key for next call
        subkeys = keys[1:]

        return self.data_generation(subkeys, replicated_state)
