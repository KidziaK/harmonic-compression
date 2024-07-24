
"""
Module contating all math functions used for both compression and decompression algorithms.
The convention for spherical coordiantes is the same as in https://mathworld.wolfram.com/SphericalCoordinates.html,
that is radial, azimuth, and zenith angle coordinates are taken as r, theta, and phi, respectively. And so:
r in [0, +inf), theta in [0, 2pi), and phi in [0, pi].

References:
-https://en.wikipedia.org/wiki/Wigner_D-matrix
-https://www.cambridge.org/core/books/hilbert-space-methods-in-signal-processing/BA54ECB490D53FF8CB176CFDCE34A962
"""

import numpy as np

from numba import njit
from scipy.special import iv, lpmv, sph_harm
import numba_scipy


__all__ = [
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    "standard_mises_fisher_harmonic_coefficients",
    "wigner_d_function",
    "reconstruct_from_harmonics"
]

@njit
def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    """
    Given a numpy array of 3D points in Cartesian coordinates, convert them to spherical coordinates.

    Args:
        points: numpy array of shape (N, 3) representing N points in Cartesian coordinates.
        
    Returns:
        numpy array of points in spherical coordinates.
    """
    # https://mathworld.wolfram.com/SphericalCoordinates.html
    result = np.zeros_like(points)
    for i in range(points.shape[0]):
        point = points[i]
        x, y, z = point[0], point[1], point[2]
        r = np.sqrt(np.dot(point, point))
        if np.isclose(r, 0):
            result[i] = np.array([0, 0, 0])
            continue
        theta =  np.mod(np.arctan2(y, x), 2 * np.pi)
        phi = np.mod(np.arccos(z / r), np.pi)
        result[i] = np.array([r, theta, phi])
    return result



@njit
def spherical_to_cartesian(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Given a numpy array of 3D points in spherical coordinates, convert them to Cartesian coordinates.

    Args:
        points: numpy array of shape (N, 3) representing N points in spherical coordinates.
        
    Returns:
        numpy array of points in Cartesian coordinates.
    """
    n = len(r)
    result = np.zeros(shape=(n, 3))
    for i in range(n):
        x = r[i] * np.cos(theta[i]) * np.sin(phi[i])
        y = r[i] * np.sin(theta[i]) * np.sin(phi[i])
        z = r[i] * np.cos(phi[i])
        result[i] = np.array([x, y, z])
    return result


@njit
def A(l: np.ndarray) -> np.ndarray:
    return np.sqrt((2 * l + 1) / (4 * np.pi))

@njit
def B(l: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    nl, nk = len(l), len(kappa)
    result = np.zeros(shape=(nl, nk))
    for i in range(nl):
        for j in range(nk):
            result[i][j] = iv(l[i] + 0.5, kappa[j]) / iv(0.5, kappa[j])
    return result

@njit
def standard_mises_fisher_harmonic_coefficients(l: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    return A(l)[:, np.newaxis] * B(l, kappa)

@njit
def factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit
def C(m: np.ndarray, l: np.ndarray) -> np.ndarray:
    nl, nm = len(l), len(m)
    result = np.zeros(shape=(nl, nm))
    for i in range(nl):
        for j in range(i, nm):
            result[i][j] = np.sqrt(factorial(l[i] - m[j]) / factorial(l[i] + m[j]))
    return result

@njit
def D(m: np.ndarray, l: np.ndarray, phi: np.ndarray) -> np.ndarray:
    nl, nm, nphi = len(l), len(m), len(phi)
    result = np.zeros(shape=(nl, nm, nphi))
    for i in range(nl):
        for j in range(nm):
            for k in range(nphi):
                result[i][j][k] = lpmv(float(m[j]), float(l[i]), np.cos(phi[k]))
    return result

@njit
def wigner_d_matrix(m: np.ndarray, l: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return C(m, l)[:, :, np.newaxis] * D(m, l, phi)

@njit
def E(m: np.ndarray, theta: np.ndarray) -> np.ndarray:
    nm, ntheta = len(m), len(theta)
    result = np.zeros(shape=(nm, ntheta), dtype=np.complex128)
    for i in range(nm):
        for j in range(ntheta):
            result[i][j] = np.exp(-1.0j * theta[j] * m[i])
    return result

@njit
def wigner_d_function(m: np.ndarray, l: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return E(m, theta)[np.newaxis, :, :] * wigner_d_matrix(m, l, phi)

def reconstruct_from_harmonics(lm_matrix: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    nl, nm = lm_matrix.shape
    ntheta, nphi = len(theta), len(phi)
    result = np.zeros(shape=(ntheta, nphi))
    for i in range(nm):
        for j in range(nl):
            for k in range(ntheta):
                for l in range(nphi):
                    if i == j: result[k][l] += np.real(sph_harm(i, j, k, l) * lm_matrix[j][i])
    return result

