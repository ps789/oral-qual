import numpy as np
import torch
from torch import Tensor, Size
import jax
import jax.numpy as jnp
import jax.random as rng
import jax_cfd.base as cfd

import math
from tqdm import tqdm

from src import utils

size = 150
Ni = 40
Nt = 200
N0 = 100
dt = 0.04
seed = 0

utils.set_seed(seed)

y = np.zeros((Ni, Nt, 2, size, size))

reynoldses = np.random.uniform(500, 1500, Ni)
# reynoldses = [600, 1000, 1500]

for ni in tqdm(range(Ni)):
    reynolds = reynoldses[ni]

    grid = cfd.grids.Grid(
                shape=(size, size),
                domain=((0, 2 * math.pi), (0, 2 * math.pi)),
            )

    bc = cfd.boundaries.periodic_boundary_conditions(2)

    forcing = cfd.forcings.simple_turbulence_forcing(
        grid=grid,
        constant_magnitude=1.0,
        constant_wavenumber=4.0,
        linear_coefficient=-0.1,
        forcing_type='kolmogorov',
    )

    dt_min = cfd.equations.stable_time_step(
        grid=grid,
        max_velocity=5.0,
        max_courant_number=0.5,
        viscosity=1 / reynolds,
    )
    
    # print("dt_min: ", dt_min)
    
    if dt_min > dt: 
        steps = 1
    else:
        steps = math.ceil(dt / dt_min)
        
    step = cfd.funcutils.repeated(
        f=cfd.equations.semi_implicit_navier_stokes(
            grid=grid,
            forcing=forcing,
            dt=dt / steps,
            density=1.0,
            viscosity=1 / reynolds,
        ),
        steps=steps,
    )
    
    key = rng.PRNGKey(42)
    
    u0, v0 = cfd.initial_conditions.filtered_velocity_field(
        key,
        grid=grid,
        maximum_velocity=3.0,
        peak_wavenumber=4.0,
    )

    yi = np.asarray(jnp.stack((u0.data, v0.data)))
    # y[ni, 0, :, :, :] = yi
    # print("y0: ", yi.shape)
    
    for ti in range(N0+Nt):
        
        u, v = cfd.initial_conditions.wrap_variables(
                    var=tuple(yi),
                    grid=grid,
                    bcs=(bc, bc),
                )
        
        u, v = step((u, v))
        
        yi = np.asarray(jnp.stack((u.data, v.data)))
        
        if ti >= N0:
            y[ni, ti-N0, :, :, :] = yi
        # print("y: ", y.shape)
        
x1 = np.linspace(0, 2 * math.pi, size)
x2 = np.linspace(0, 2 * math.pi, size)
X1, X2 = np.meshgrid(x1, x2)
X1 = np.transpose(X1)
X2 = np.transpose(X2)
coords = np.stack((X1.ravel(), X2.ravel()), axis=1)
        
np.savez("data/data_cplx_Re_500_1500_150x150_seed_0.npz", y=y, u=reynoldses, x=coords, dt=dt)