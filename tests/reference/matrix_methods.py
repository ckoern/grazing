"""Matrix formalisms for planar stacks, written independently of `grazing.core.physics.kernel`.

These are straightforward but not numerically stable for thick absorbing stacks; they are
intended for small to moderate test stacks only.

Conventions match the library: grazing angle θ, n = 1 − δ + iβ, media ordered from the ambient
(index 0) to the substrate (index M − 1), s coefficients for the electric field, p coefficients
for the magnetic field.
"""

import mpmath
import numpy as np


def kz_over_k(n: np.ndarray, theta: float) -> np.ndarray:
    """Normalized perpendicular wavevector, Im ≥ 0, for one evaluation point."""
    n0 = n[0]
    q = np.sqrt((n0 * np.sin(theta)) ** 2 + (n - n0) * (n + n0))
    result: np.ndarray = np.where(q.imag < 0, -q, q)
    return result


def abeles(
    n: np.ndarray, thickness: np.ndarray, theta: float, wavelength: float, polarization: str
) -> complex:
    """Reflection amplitude from the Abelès characteristic matrix (sharp interfaces only).

    Each layer contributes the matrix [[cos φ, −i sin φ / q], [−i q sin φ, cos φ]] with
    φ = k_z d, q = k_z (s) or q = k_z / n² (p), relating tangential fields at its two faces.
    """
    k = 2 * np.pi / wavelength
    kz = k * kz_over_k(n, theta)
    q = kz if polarization == "s" else kz / n**2
    m = np.eye(2, dtype=complex)
    for j, d in enumerate(thickness, start=1):
        phi = kz[j] * d
        layer = np.array(
            [[np.cos(phi), -1j * np.sin(phi) / q[j]], [-1j * q[j] * np.sin(phi), np.cos(phi)]]
        )
        m = m @ layer
    q0, qs = q[0], q[-1]
    num = q0 * m[0, 0] + q0 * qs * m[0, 1] - m[1, 0] - qs * m[1, 1]
    den = q0 * m[0, 0] + q0 * qs * m[0, 1] + m[1, 0] + qs * m[1, 1]
    return complex(num / den)


def transfer_matrix(
    n: np.ndarray,
    thickness: np.ndarray,
    roughness: np.ndarray,
    theta: float,
    wavelength: float,
    polarization: str,
) -> tuple[complex, complex]:
    """Reflection and transmission amplitudes from interface and propagation matrices.

    Interfaces with Névot–Croce roughness use r·exp(−2 k_z,j k_z,j+1 σ²) and
    t·exp((k_z,j − k_z,j+1)² σ² / 2), with the interface matrix (1/t)[[1, r], [r, 1]].
    """
    k = 2 * np.pi / wavelength
    kz = k * kz_over_k(n, theta)
    eps = n**2
    m = np.eye(2, dtype=complex)
    for j in range(len(n) - 1):
        a, b = kz[j], kz[j + 1]
        if polarization == "s":
            r = (a - b) / (a + b)
        else:
            r = (eps[j + 1] * a - eps[j] * b) / (eps[j + 1] * a + eps[j] * b)
        t = 1 + r
        s2 = roughness[j] ** 2
        r = r * np.exp(-2 * a * b * s2)
        t = t * np.exp(0.5 * (a - b) ** 2 * s2)
        m = m @ (np.array([[1, r], [r, 1]], dtype=complex) / t)
        if j + 1 < len(n) - 1:
            phi = kz[j + 1] * thickness[j]
            m = m @ np.diag([np.exp(-1j * phi), np.exp(1j * phi)])
    return complex(m[1, 0] / m[0, 0]), complex(1 / m[0, 0])


def transfer_matrix_mp(
    n: np.ndarray,
    thickness: np.ndarray,
    roughness: np.ndarray,
    theta: float,
    wavelength: float,
    polarization: str,
    digits: int = 50,
) -> tuple[complex, complex]:
    """Arbitrary-precision version of `transfer_matrix`."""
    with mpmath.workdps(digits):
        nn = [mpmath.mpc(complex(v)) for v in n]
        th = mpmath.mpf(float(theta))
        k = 2 * mpmath.pi / mpmath.mpf(float(wavelength))
        kz = []
        for v in nn:
            q = mpmath.sqrt((nn[0] * mpmath.sin(th)) ** 2 + (v - nn[0]) * (v + nn[0]))
            kz.append(k * (-q if q.imag < 0 else q))
        m = mpmath.eye(2)
        for j in range(len(nn) - 1):
            a, b = kz[j], kz[j + 1]
            if polarization == "s":
                r = (a - b) / (a + b)
            else:
                e1, e0 = nn[j + 1] ** 2, nn[j] ** 2
                r = (e1 * a - e0 * b) / (e1 * a + e0 * b)
            t = 1 + r
            s2 = mpmath.mpf(float(roughness[j])) ** 2
            r = r * mpmath.exp(-2 * a * b * s2)
            t = t * mpmath.exp((a - b) ** 2 * s2 / 2)
            m = m * (mpmath.matrix([[1, r], [r, 1]]) / t)
            if j + 1 < len(nn) - 1:
                phi = kz[j + 1] * mpmath.mpf(float(thickness[j]))
                m = m * mpmath.matrix([[mpmath.exp(-1j * phi), 0], [0, mpmath.exp(1j * phi)]])
        return complex(m[1, 0] / m[0, 0]), complex(1 / m[0, 0])
