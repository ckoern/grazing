"""Convolution of curves with an instrumental resolution function."""

import numpy as np
from numpy.typing import NDArray

__all__ = ["extended_grid", "gaussian_smear"]

_RANGE = 4.0
"""Half width of the Gaussian kernel in units of its standard deviation."""


def extended_grid(
    lo: float, hi: float, points: int, sigma: float
) -> tuple[NDArray[np.float64], slice]:
    """Return a uniform grid that extends a range by the reach of a Gaussian kernel.

    Args:
        lo: Lower end of the range of interest.
        hi: Upper end of the range of interest.
        points: Number of points of the original grid on [lo, hi].
        sigma: Standard deviation of the Gaussian, in the units of the grid.

    Returns:
        The extended grid (spacing at most σ/3 and at most the original spacing) and the
        slice of that grid covering [lo, hi].
    """
    step = min((hi - lo) / (points - 1), sigma / 3.0)
    inner = int(np.ceil((hi - lo) / step)) + 1
    pad = int(np.ceil(_RANGE * sigma / step))
    grid = lo + step * np.arange(-pad, inner + pad, dtype=np.float64)
    return grid, slice(pad, pad + inner)


def gaussian_smear(y: NDArray[np.float64], step: float, sigma: float) -> NDArray[np.float64]:
    """Convolve a curve on a uniform grid with a normalized Gaussian.

    Values beyond the ends of the curve are taken equal to the end values.

    Args:
        y: Curve values on a uniform grid.
        step: Grid spacing.
        sigma: Standard deviation of the Gaussian, in the units of the grid. Zero returns a copy.

    Returns:
        The convolved curve, same shape as `y`.
    """
    if sigma <= 0.0:
        return np.array(y, dtype=np.float64)
    half = int(np.ceil(_RANGE * sigma / step))
    offsets = step * np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    return np.convolve(padded, kernel, mode="valid")
