import numpy as np
import scipy
from numba import njit, prange
from scipy.special import iv, lpmv
import spherical
import quaternionic
import math
import numba_scipy

from utility.e57 import load_e57

l_max = 3
wigner = spherical.Wigner(l_max + 1) # We need +1 for recurrence relation for derivative of wigner d

@njit(parallel=True)
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


@njit(parallel=True)
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

@njit
def B(l: int):
    return np.sqrt((2 * l + 1) / 4 / np.pi)

def g(l, kappa):
    """
    Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
    Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
    """
    return (B(l) * iv(l + 0.5, kappa) / iv(0.5, kappa)).astype(np.complex128)

@njit
def g_(l, kappa):
    """
    Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
    Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
    """
    return B(l) * iv(l + 0.5, kappa) / iv(0.5, kappa)

@njit(parallel=True)
def harmonic_f(wigner_D, f_tilda, ell_max: int):
    num_points = wigner_D.shape[0]
    f = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)
    for i in prange(num_points):
        for l in range(ell_max + 1):
            for m in range(l + 1):
                f[i, l, m] = f_tilda[i, l] * wigner_D[i, spherical.WignerDindex(l, 0, m, 0, ell_max)]
    return f

def compress(spherical_coords):
    kappa = spherical_coords[:, 0].astype(np.float64)
    theta = spherical_coords[:, 1].astype(np.float64)
    phi = spherical_coords[:, 2].astype(np.float64)

    l = np.arange(0, l_max + 1)

    l_grid, kappa_grid = np.meshgrid(l, kappa)

    f_tilda = g(l_grid, kappa_grid)

    q = quaternionic.array.from_euler_angles(theta, phi, np.zeros_like(theta))

    D = wigner.D(q)

    return harmonic_f(D, f_tilda, l_max)

@njit(parallel=True)
def N(l: int):
    return np.sqrt((2 * l + 1) / (4 * np.pi))

@njit(parallel=True)
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

@njit(parallel=True)
def df_dkappa(wigner_D, kappa, ell_max: int):
    num_points = wigner_D.shape[0]
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)
    sqrt_2pi = np.sqrt(2 * np.pi)
    for i in prange(num_points):
        k = kappa[i]
        sqrt_k = np.sqrt(k)
        csch_k = 1 / np.sinh(k)
        coth_k = np.cosh(k) / np.sinh(k)
        for l in range(ell_max + 1):
            first_term = csch_k * iv(k, l + 0.5) / (4 * sqrt_2pi * sqrt_k)
            second_term = sqrt_k * csch_k * (iv(k, l - 0.5) + iv(k, l + 1.5)) / (4 * sqrt_2pi)
            third_term = sqrt_k * coth_k * csch_k * iv(k, l + 0.5) / (2 * sqrt_2pi)
            dg = first_term + second_term - third_term
            for m in range(l + 1):
                df[i, l, m] = wigner_D[i, spherical.WignerDindex(l, 0, m, 0, ell_max)] * dg
    return df

@njit
def factorial(x):
    if x == 0:
        return 1
    return x * factorial(x - 1)

@njit(parallel=True)
def df_dtheta(d, kappa, theta, phi, ell_max: int):
    num_points = len(theta)
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)

    for i in prange(num_points):
        for l in range(ell_max + 1):
            for m in range(l + 1):
                if m == 0:
                    dd = -np.sqrt(l * (l+1)) * d[i, spherical.WignerDindex(l, 0, m + 1, 0, ell_max)]
                    dD = np.exp(-1j * m * phi[i]) * dd
                    df[i, l, m] = dD * g_(l, kappa[i])
                else:
                    constant = 0.5 * np.sqrt(factorial(l - m)/factorial(l + m))
                    legendre_1 = (l + m) * (l - m + 1) * lpmv(float(m - 1), float(l), np.cos(theta[i]))
                    legendre_2 = lpmv(float(m - 1), float(l), np.cos(theta[i]))
                    dD = np.exp(-1j * m * phi[i]) * constant * (legendre_1 - legendre_2)
                    df[i, l, m] = dD * g_(l, kappa[i])

    return df

@njit(parallel=True)
def df_dphi(d, kappa, theta, phi, ell_max: int):
    num_points = len(theta)
    df = np.zeros(shape=(num_points, ell_max + 1, ell_max + 1), dtype=np.complex128)

    for i in prange(num_points):
        for l in range(ell_max + 1):
            for m in range(l + 1):
                dml = d[i, spherical.WignerDindex(l, 0, m, 0, ell_max)]
                df[i, l, m] = - 1j * m * np.exp(-1j * m * phi[i]) * dml * g_(l, kappa[i])

    return df

step = 0

def loss(spherical_coords_flat, true_coefficients):
    n = len(spherical_coords_flat)
    global step
    spherical_coords = spherical_coords_flat.reshape((3, n // 3)).T
    predicted_coefficients = compress(spherical_coords)
    F_tilda = predicted_coefficients.sum(axis=0)
    F = true_coefficients.sum(axis=0)
    diff = F_tilda - F
    L = np.real(np.conjugate(diff) * diff).sum()
    # print(f"Loss at step {step} = {L}")
    step += 1
    return L
#
# def grad_loss(spherical_coords_flat, true_coefficients):
#     n = len(spherical_coords_flat)
#     spherical_coords = spherical_coords_flat.reshape((3, n // 3)).T
#     predicted_coefficients = compress(spherical_coords)
#     F_tilda = predicted_coefficients.sum(axis=0)
#     F = true_coefficients.sum(axis=0)
#     diff = F_tilda - F
#
#     kappa = spherical_coords[:, 0].astype(np.float64)
#     theta = spherical_coords[:, 1].astype(np.float64)
#     phi = spherical_coords[:, 2].astype(np.float64)
#
#     q = quaternionic.array.from_euler_angles(theta, phi, np.zeros_like(theta))
#
#     D = wigner.D(q)
#     d_list = []
#     for i in range(len(theta)):
#         d_list.append(wigner.d(np.exp(1j * theta[i])))
#     d = np.array(d_list)
#
#     dkappa = df_dkappa(D, kappa, l_max)
#     dtheta = df_dtheta(d, kappa, theta, phi, l_max)
#     dphi = df_dphi(d, kappa, theta, phi, l_max)
#
#     dL_dkappa = np.real(np.conjugate(dkappa) * diff).mean(axis=(1, 2))
#     dL_dtheta = np.real(np.conjugate(dtheta) * diff).mean(axis=(1, 2))
#     dL_dphi = np.real(np.conjugate(dphi) * diff).mean(axis=(1, 2))
#
#     return -np.hstack([dL_dkappa, dL_dtheta, dL_dphi])

if __name__ == "__main__":
    import time

    points = load_e57("../data/Bunny.e57")[100:150]
    # points = load_e57("../data/cube.e57")

    radius_max = np.linalg.norm(points, axis=1, keepdims=True).max()
    points = points / radius_max
    spherical_coords = cartesian_to_spherical(points)

    kappa = spherical_coords[:, 0].astype(np.float64)
    theta = spherical_coords[:, 1].astype(np.float64)
    phi = spherical_coords[:, 2].astype(np.float64)

    error_theta = 0.1

    kappa_start = 0.5 * np.ones_like(spherical_coords[:, 0])
    theta_start = np.clip(spherical_coords[:, 1] + np.random.uniform(-error_theta, error_theta), 0.0, 2*np.pi)
    phi_start = np.clip(spherical_coords[:, 2] + np.random.uniform(-0.1, 0.1), 0.0, np.pi)
    # theta_start = np.random.uniform(0, 2 * np.pi, size=len(theta))
    # phi_start = np.random.uniform(0, np.pi, size=len(phi))

    st = time.time()
    harmonic_coefficients = compress(spherical_coords).astype(np.complex64)
    end = time.time()

    print(f"compression took: {end - st} seconds")

    n = len(kappa_start)
    bounds_kappa = np.array([[0, 1] for _ in range(n)])
    bounds_theta = np.array([[theta[i] - error_theta, theta[i] + error_theta] for i in range(n)])
    bounds_theta =  np.clip(bounds_theta, 0, 2 * np.pi)

    bounds_phi = np.array([[0, np.pi] for _ in range(n)])

    bounds = np.concatenate([bounds_kappa, bounds_theta, bounds_phi])

    reconstruction = scipy.optimize.minimize(loss, np.hstack([kappa_start, theta_start, phi_start]), harmonic_coefficients, bounds=bounds
                                             # , jac=grad_loss
                                             )
    kappa_reconstructed = reconstruction.x[:n]
    theta_reconstructed = reconstruction.x[n:2*n]
    phi_reconstructed = reconstruction.x[2*n:3*n]

    from sklearn.metrics import mean_squared_error

    print(f"MSE(kappa) = {mean_squared_error(kappa, kappa_reconstructed)}")
    print(f"MSE(theta) = {mean_squared_error(theta, theta_reconstructed)}")
    print(f"MSE(phi) = {mean_squared_error(phi, phi_reconstructed)}")

    print(f"PSNR(kappa) = {-10 * math.log(mean_squared_error(kappa, kappa_reconstructed))}")
    print(f"PSNR(theta) = {20*np.log10(2 * np.pi) -10 * math.log(mean_squared_error(theta, theta_reconstructed))}")
    print(f"PSNR(phi) = {20*np.log10(2 * np.pi)-10 * math.log(mean_squared_error(phi, phi_reconstructed))}")

    st = time.time()
    reconstructed_points = spherical_to_cartesian(kappa, theta, phi)
    end = time.time()

    print(f"reconstruction took: {end - st} seconds")

    # print(points)
    # print(reconstructed_points)