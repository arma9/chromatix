#!/usr/bin/env python3
"""
Example usage of the system_prop_lens_prop optical system.

This demonstrates how to create and use an optical system that:
1. Creates a Gaussian plane wave source
2. Propagates it by distance z1
3. Applies a thin lens with focal length f
4. Propagates it by distance z2
"""

import numpy as np
import jax.numpy as jnp
from chromatix.systems import system_prop_lens_prop

# Define system parameters
shape = (256, 256)  # Field grid size
dx = 1e-6  # 1 micron pixel spacing
spectrum = 632.8e-9  # Red laser wavelength (HeNe)
spectral_density = 1.0
waist = 50e-6  # 50 micron beam waist
z1 = 1e-3  # 1 mm propagation before lens
z2 = 1e-3  # 1 mm propagation after lens
f = 5e-3  # 5 mm focal length lens
n = 1.0  # Air
power = 1.0
NA = 0.5  # Numerical aperture

# Create the optical system
optical_system = system_prop_lens_prop(
    shape=shape,
    dx=dx,
    spectrum=spectrum,
    spectral_density=spectral_density,
    waist=waist,
    z1=z1,
    z2=z2,
    f=f,
    n=n,
    power=power,
    NA=NA
)

# Initialize the system (this is required for JAX/Flax modules)
import jax
key = jax.random.PRNGKey(0)
params = optical_system.init(key)

# Run the optical system
output_field = optical_system.apply(params)

print(f"Input parameters:")
print(f"  Grid shape: {shape}")
print(f"  Pixel spacing: {dx*1e6:.1f} μm")
print(f"  Wavelength: {spectrum*1e9:.1f} nm")
print(f"  Beam waist: {waist*1e6:.1f} μm")
print(f"  Propagation z1: {z1*1e3:.1f} mm")
print(f"  Lens focal length: {f*1e3:.1f} mm")
print(f"  Propagation z2: {z2*1e3:.1f} mm")
print(f"  Lens NA: {NA}")

print(f"\nOutput field:")
print(f"  Shape: {output_field.u.shape}")
print(f"  Peak intensity: {np.max(output_field.intensity):.3e}")
print(f"  Total power: {np.sum(output_field.intensity) * (dx**2):.3e}")

# The output_field is now a Field object containing the result after:
# Gaussian source -> propagate z1 -> lens -> propagate 
