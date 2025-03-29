import numpy as np
import scipy
from numba import njit, prange
from scipy.special import iv, lpmv
import spherical
import quaternionic
import math
import numba_scipy

from utility.e57 import load_e57

l_max = 10
wigner = spherical.Wigner(l_max + 1) # We need +1 for recurrence relation for derivative of wigner d

@njit(parallel=True, cache=True)
def cartesian_to_spherical(points):
    """
    Given a numpy array of 3D points in Cartesian coordinates, convert them to spherical coordinates.

    Args:
        points: numpy array of shape (N, 3) representing N points in Cartesian coordinates.

    Returns:
        numpy array of points in spherical coordinates.
    """
    # https://mathworld.wolfram.com/SphericalCoordinates.html
    result = np.zeros_like(points, dtype=np.float32)
    for i in prange(points.shape[0]):
        point = points[i]
        x, y, z = point[0], point[1], point[2]
        r = np.sqrt(np.dot(point, point))
        if np.isclose(r, 0):
            result[i] = np.array([0, 0, 0])
            continue
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)
        phi = np.mod(np.arccos(z / r), np.pi)
        result[i] = np.array([r, theta, phi])
    return result


@njit(parallel=True, cache=True)
def spherical_to_cartesian(r, theta, phi):
    """
    Given a numpy array of 3D points in spherical coordinates, convert them to Cartesian coordinates.

    Args:
        points: numpy array of shape (N, 3) representing N points in spherical coordinates.

    Returns:
        numpy array of points in Cartesian coordinates.
    """
    n = len(r)
    result = np.zeros(shape=(n, 3))
    for i in prange(n):
        x = r[i] * np.cos(theta[i]) * np.sin(phi[i])
        y = r[i] * np.sin(theta[i]) * np.sin(phi[i])
        z = r[i] * np.cos(phi[i])
        result[i] = np.array([x, y, z])
    return result

@njit(parallel=True, cache=True)
def B(l: int):
    return np.sqrt((2 * l + 1) / 4 / np.pi)

def g(l, kappa):
    """
    Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
    Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
    """
    return (B(l) * iv(l + 0.5, kappa) / iv(0.5, kappa)).astype(np.complex128)

@njit(parallel=True, cache=True)
def harmonic_f(wigner_D, f_tilda, ell_max: int):
    num_points = wigner_D.shape[0]
    f = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)
    for i in prange(num_points):
        for l in range(ell_max + 1):
            for m in range(l + 1):
                f[i, l, m] = f_tilda[i, l] * wigner_D[i, spherical.WignerDindex(l, 0, m, 0, ell_max)]
    return f

def compress(spherical_coords):
    kappa = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]
    phi = spherical_coords[:, 2]

    l = np.arange(0, l_max + 1)

    l_grid, kappa_grid = np.meshgrid(l, kappa)

    f_tilda = g(l_grid, kappa_grid)

    q = quaternionic.array.from_euler_angles(theta, phi, np.zeros_like(theta))

    D = wigner.D(q)

    return harmonic_f(D, f_tilda, l_max)

@njit(parallel=True, cache=True)
def N(l: int):
    return np.sqrt((2 * l + 1) / (4 * np.pi))

@njit(parallel=True, cache=True)
def dg_dkappa(kappa):
    n = len(kappa)
    dg = np.zeros(shape=(n, l_max + 1), dtype=np.complex128)
    sqrt_2pi = np.sqrt(2 * np.pi)
    for i in prange(n):
        k = kappa[i]
        sqrt_k = np.sqrt(k)
        csch_k = 1 / np.sinh(k)
        coth_k = np.cosh(k) / np.sinh(k)
        for l in range(l_max + 1):
            first_term = csch_k * iv(k, l + 0.5) / (4 * sqrt_2pi * sqrt_k)
            second_term = sqrt_k * csch_k * (iv(k, l - 0.5) + iv(k, l + 1.5)) / (4 * sqrt_2pi)
            third_term = sqrt_k * coth_k * csch_k * iv(k, l + 0.5) / (2 * sqrt_2pi)
            dg[i, l] = first_term + second_term - third_term
    return dg

@njit(parallel=True, cache=True)
def df_dkappa(wigner_D, kappa, ell_max: int):
    num_points = wigner_D.shape[0]
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)
    for i in prange(num_points):
        for l in range(ell_max + 1):
            for m in range(l + 1):
                df[i, l, m] = wigner_D[i, spherical.WignerDindex(l, 0, m, 0, ell_max)] * dg_dkappa(kappa)
    return df

@njit(parallel=True, cache=True)
def df_dtheta(kappa, theta, phi, ell_max: int):
    num_points = d.shape[0]
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)

    for i in prange(num_points):
        d = wigner.d(np.exp(1j * theta[i]))
        for l in range(ell_max + 1):
            for m in range(l + 1):
                if m == 0:
                    dd = -np.sqrt(l * (l+1)) * d[spherical.WignerDindex(l, 0, m + 1, 0, ell_max)]
                    dD = np.exp(-1j * m * phi[i]) * dd
                    df[i, l, m] = dD * g(l, kappa[i])
                else:
                    constant = 0.5 * np.sqrt(math.factorial(l - m)/math.factorial(l + m))
                    legendre_1 = (l + m) * (l - m + 1) * lpmv(m - 1, l, theta[i])
                    legendre_2 = lpmv(m + 1, l, theta[i])
                    dD = np.exp(-1j * m * phi[i]) * constant * (legendre_1 - legendre_2)
                    df[i, l, m] = dD * g(l, kappa[i])

    return df

@njit(parallel=True, cache=True)
def df_dphi(theta, phi, ell_max: int):
    num_points = d.shape[0]
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)

    for i in prange(num_points):
        d = wigner.d(np.exp(1j * theta[i]))
        for l in range(ell_max + 1):
            for m in range(l + 1):
                dml = d[spherical.WignerDindex(l, 0, m, 0, ell_max)]
                df[i, l, m] = - 1j * m * np.exp(-1j * m * phi[i]) * dml

    return df

def loss(spherical_coords_flat, true_coefficients):
    n = len(spherical_coords_flat)
    spherical_coords = spherical_coords_flat.reshape((3, n // 3)).T
    predicted_coefficients = compress(spherical_coords)
    F_tilda = predicted_coefficients.sum(axis=0)
    F = true_coefficients.sum(axis=0)
    diff = F_tilda - F
    return np.real(np.conjugate(diff) * diff).sum()

def grad_loss(spherical_coords_flat, true_coefficients):
    n = len(spherical_coords_flat)
    spherical_coords = spherical_coords_flat.reshape((3, n // 3)).T
    predicted_coefficients = compress(spherical_coords)
    F_tilda = predicted_coefficients.sum(axis=0)
    F = true_coefficients.sum(axis=0)
    diff = F_tilda - F

    kappa = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]
    phi = spherical_coords[:, 2]

    q = quaternionic.array.from_euler_angles(theta, phi, np.zeros_like(theta))

    D = wigner.D(q)

    dkappa = df_dkappa(D, kappa, l_max)
    dtheta = df_dtheta(kappa, theta, phi, l_max)
    dphi = df_dphi(theta, phi, l_max)

    dL_dkappa = 2 * np.real(np.conjugate(dkappa) * diff).sum(axis=(1, 2))
    dL_dtheta = 2 * np.real(np.conjugate(dtheta) * diff).sum(axis=(1, 2))
    dL_dphi = 2 * np.real(np.conjugate(dphi) * diff).sum(axis=(1, 2))

    return np.array([dL_dkappa, dL_dtheta, dL_dphi])

if __name__ == "__main__":
    import time

    points = load_e57("../data/cube.e57")

    radius = np.linalg.norm(points, axis=1, keepdims=True)
    radius_scaled = (np.arctan(radius) / np.pi + 0.5) * 40 + 10

    points = points / radius * radius_scaled

    spherical_coords = cartesian_to_spherical(points)

    kappa_start = np.ones_like(spherical_coords[:, 0])
    theta_start = spherical_coords[:, 1]
    phi_start = spherical_coords[:, 2]

    st = time.time()
    harmonic_coefficients = compress(spherical_coords)
    end = time.time()

    print(f"compression took: {end - st} seconds")

    print(harmonic_coefficients)

    reconstruction = scipy.optimize.minimize(loss, np.hstack([kappa_start, theta_start, phi_start]), harmonic_coefficients, jac=grad_loss)
