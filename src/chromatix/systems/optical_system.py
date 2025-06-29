from __future__ import annotations

from typing import Any, Callable, Sequence

from flax import linen as nn
from jax import Array

from ..field import Field


class OpticalSystem(nn.Module):
    """
    Combines a sequence of optical elements into a single ``Module``.

    Takes a sequence of functions or ``Module``s (any ``Callable``) and calls
    them in sequence, assuming each element of the sequence only accepts a
    ``Field`` as input and returns a ``Field`` as output, with the exception of
    the first element of the sequence, which can take any arguments necessary
    (e.g. to allow an element from ``chromatix.elements.sources`` to initialize
    a ``Field``) and the last element of the sequence, which may return an
    ``Array``. This is intended to mirror the style of deep learning libraries
    that describe a neural network as a sequence of layers, allowing for an
    optical system to be described conveniently as a list of elements.

    Attributes:
        elements: A sequence of optical elements describing the system.
    """

    elements: Sequence[Callable]

    @nn.compact
    def __call__(self, *args: Any, **kwargs: Any) -> Field | Array:
        """Returns the result of calling all elements in sequence."""
        field = self.elements[0](*args, **kwargs)  # allow field to be initialized
        for element in self.elements[1:]:
            field = element(field)
        return field


def system_prop_lens_prop(
    shape: tuple[int, int],
    dx: float,
    spectrum: float,
    spectral_density: float = 1.0,
    waist: float = 1e-3,
    z1: float = 1e-3,
    z2: float = 1e-3,
    f: float = 1e-3,
    n: float = 1.0,
    power: float = 1.0,
    NA: float | None = None,
) -> OpticalSystem:
    """
    Creates an optical system with Gaussian source -> propagate -> lens -> propagate.
    
    This system:
    1. Creates a Gaussian plane wave source
    2. Propagates it by distance z1
    3. Applies a thin lens with focal length f
    4. Propagates it by distance z2
    
    Args:
        shape: The shape (height, width) of the field grid.
        dx: The spatial sampling interval.
        spectrum: The wavelength(s) to simulate.
        spectral_density: The spectral weight(s). Defaults to 1.0.
        waist: The waist size of the Gaussian beam. Defaults to 1e-3.
        z1: First propagation distance (before lens). Defaults to 1e-3.
        z2: Second propagation distance (after lens). Defaults to 1e-3.
        f: Focal length of the lens. Defaults to 1e-3.
        n: Refractive index of the medium. Defaults to 1.0.
        power: Total power of the source. Defaults to 1.0.
        NA: Numerical aperture of the lens. If None, no pupil is applied.
        
    Returns:
        OpticalSystem: The configured optical system.
    """
    from ..elements import GaussianPlaneWave, Propagate, ThinLens
    
    return OpticalSystem(
        elements=[
            GaussianPlaneWave(
                shape=shape,
                dx=dx,
                spectrum=spectrum,
                spectral_density=spectral_density,
                waist=waist,
                power=power,
            ),
            Propagate(z=z1, n=n),
            ThinLens(f=f, n=n, NA=NA),
            Propagate(z=z2, n=n),
        ]
    )
