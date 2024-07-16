import numpy as np

from numba import njit
from scipy.special import iv, lpmv, factorial
from typing import Tuple
from functools import cache

# https://mathworld.wolfram.com/SphericalCoordinates.html
@njit
def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    result = np.zeros_like(points)
    for i in range(points.shape[0]):
        point = points[i]
        x, y, z = point[0], point[1], point[2]
        r = np.sqrt(np.dot(point, point))
        if np.isclose(r, 0):
            result[i] = np.array([0, 0, 0])
            continue
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r)
        result[i] = np.array([r, theta, phi])
    return result

@njit
def spherical_to_cartesian(kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    n = len(points)
    result = np.zeros(shape=(n, 3))
    for i in range(n):
        x = kappa[i] * np.cos(theta[i]) * np.sin(phi[i])
        y = kappa[i] * np.sin(theta[i]) * np.sin(phi[i])
        z = kappa[i] * np.cos(phi[i])
        result[i] = np.array([x, y, z])
    return result

@njit
def get_euler_angles(theta_array: np.ndarray, phi_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta, phi, gamma = np.zeros_like(phi_array), np.zeros_like(phi_array), np.zeros_like(phi_array)
    for i in range(phi_array.shape[0]):
        theta = theta_array[i]
        phi = phi_array[i]

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        r = np.array([
            [cos_theta * cos_phi, -sin_theta, cos_theta * sin_phi],
            [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi],
            [-sin_phi, 0, cos_phi]
        ])

        theta[i] = np.arctan2(r[1][2], r[0][2])
        phi[i] = np.arctan2(np.sqrt(1 - r[2][2] * r[2][2]), r[2][2])
        gamma[i] = np.arctan2(r[2][1], -r[2][0])
    
    return theta, phi, gamma

@cache
def B(l: int):
    return np.sqrt((2 * l + 1) / 4 / np.pi)

def harmonic_tilda(l: int, kappa: np.ndarray) -> np.ndarray:
    """ 
    Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
    Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
    """
    return B(l) * iv(l + 0.5, kappa) / iv(0.5, kappa)

@cache
def A(m: int, l: int):
    return np.sqrt(factorial(l - m) / factorial(l + m))

# https://en.wikipedia.org/wiki/Wigner_D-matrix
# https://www.cambridge.org/core/books/hilbert-space-methods-in-signal-processing/BA54ECB490D53FF8CB176CFDCE34A962
def wigner_d_matrix(m: int, l: int, phi: np.ndarray) -> np.ndarray:
    return A(m, l) * lpmv(m, l, np.cos(phi))

def wigner_d_function(m: int, l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.exp(-1.0j * theta * m) * wigner_d_matrix(m, l, phi)

def harmonic_f(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.mean(harmonic_tilda(l, kappa) * wigner_d_function(m, l, theta, phi))

def compress(point_cloud: np.ndarray, l_max: int) -> dict:
    spherical = cartesian_to_spherical(point_cloud)
    coefficients = {}

    kappa = spherical[:, 0]
    theta = spherical[:, 1]
    phi = spherical[:, 2]

    for l in range(l_max):
        for m in range(l+1):
            coefficients[(m, l)] = harmonic_f(m, l, kappa, theta, phi)

    return coefficients

def d_d_matrix_d_phi(m: int, l: int, phi: np.ndarray) -> np.ndarray:
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi) + 1e-5
    return A(m, l) * (1 / sin_phi) * (lpmv(m + 1, l, cos_phi) + m * cos_phi * lpmv(m, l, cos_phi))

def d_func_d_phi(m: int, l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.exp(-1.0j * theta * m) * d_d_matrix_d_phi(m, l, phi)

def d_func_d_theta(m: int, l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return -1.0j * m * wigner_d_function(l, m, theta, phi)

@cache
def N(m: int, l: int):
    return A(m, l) * B(l)

def c(kappa: np.ndarray) -> np.ndarray:
    return np.sqrt(kappa) / np.power(2 * np.pi, 3/2) * iv(0.5, kappa)

@njit
def d_c_d_kappa(kappa: np.ndarray) -> np.ndarray:
    return (np.sinh(kappa) - kappa * np.cosh(kappa)) / np.pi / 4 / np.sinh(kappa) / np.sinh(kappa)

@njit
def IK(kappa: np.ndarray) -> np.ndarray:
    return np.sqrt(2 * np.pi / kappa) 

@njit
def d_IK_d_kappa(kappa: np.ndarray) -> np.ndarray:
    return -np.sqrt(2 * np.pi) * np.power(kappa, 1.5) / 2

def I(kappa: np.ndarray, l: int) -> np.ndarray:
    return IK(kappa) * iv(l + 0.5, kappa)

def d_iv_d_kappa(kappa: np.ndarray, l: int) -> np.ndarray:
    return (iv(l - 0.5, kappa) + iv(l + 1.5, kappa)) / 2

def d_I_d_kappa(kappa: np.ndarray, l: int) -> np.ndarray:
    return d_IK_d_kappa(kappa) * iv(l + 0.5, kappa) + IK(kappa) * d_iv_d_kappa(kappa, l)

def d_g_tilda_d_kappa(l: int, m: int, kappa: np.ndarray) -> np.ndarray:
    return 2 * np.pi * N(m, l) * (d_c_d_kappa(kappa) * I(kappa, l) / c(kappa) + d_I_d_kappa(kappa, l) * c(kappa))

def d_g_d_kappa(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return wigner_d_function(l, m, theta, phi) * d_g_tilda_d_kappa(l, m, kappa)

def d_g_d_theta(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return d_func_d_theta(m, l, theta, phi) * harmonic_tilda(l, kappa)

def d_g_d_phi(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return d_func_d_phi(m, l, theta, phi) * harmonic_tilda(l, kappa)

def L(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
    loss = 0
    for (m, l), f in coeffs.items():
        g = harmonic_f(m, l, kappa, theta, phi)
        inner = np.abs(np.sum(g) - f)
        loss += inner * inner
    return loss

def d_L_d_theta(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
    d_loss = 0
    for (m, l), f in coeffs.items():
        g = harmonic_f(m, l, kappa, theta, phi)
        dg = d_g_d_theta(m, l, kappa, theta, phi)
        diff_g_f = g - f
        # We add np.real to get rid of numerical issues.
        # TODO fix this. The sum over i is an inner sum, not the outer sum
        d_loss += np.real(np.sum(dg * np.conjugate(diff_g_f) + np.conjugate(dg) * diff_g_f))
    return d_loss

def d_L_d_phi(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
    d_loss = 0
    for (m, l), f in coeffs.items():
        g = harmonic_f(m, l, kappa, theta, phi)
        dg = d_g_d_phi(m, l, kappa, theta, phi)
        diff_g_f = g - f
        # We add np.real to get rid of numerical issues.
        d_loss += np.real(np.sum(dg * np.conjugate(diff_g_f) + np.conjugate(dg) * diff_g_f))
    return d_loss

def d_L_d_kappa(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
    d_loss = 0
    for (m, l), f in coeffs.items():
        g = harmonic_f(m, l, kappa, theta, phi)
        dg = d_g_d_kappa(m, l, kappa, theta, phi)
        diff_g_f = g - f
        # We add np.real to get rid of numerical issues.
        d_loss += np.real(dg * np.conjugate(diff_g_f) + np.conjugate(dg) * diff_g_f)
    return d_loss

def grad_step(eta: np.ndarray, coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dk, dt, dp = d_L_d_kappa(coeffs, kappa, theta, phi), d_L_d_theta(coeffs, kappa, theta, phi), d_L_d_phi(coeffs, kappa, theta, phi)
    # dk = dt / dt.max()
    # dk = dp / dp.max()
    kappa = kappa - eta[0] * dk
    theta = theta - eta[1] * dt
    phi = phi - eta[2] * dp
    print(L(coeffs, kappa, theta, phi))
    # print(np.sum(dk < 0))
    # print(dk.max(), dt.max(),dp.max())

    return kappa, theta, phi

def decompress(n: int, eta: np.ndarray, steps: int, coeffs: dict) -> np.ndarray:
    uniform_points = np.random.standard_normal(size=(n, 3))
    uniform_points = uniform_points / np.linalg.norm(uniform_points, axis=1)[:, np.newaxis]
    spherical = cartesian_to_spherical(uniform_points)

    kappa = spherical[:, 0]
    theta = spherical[:, 1]
    phi = spherical[:, 2]

    for i in range(steps):
        kappa, theta, phi = grad_step(eta, coeffs, kappa, theta, phi)

    return spherical_to_cartesian(kappa, theta, phi)

def test_cartesian_to_spherical():
    cartesian = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [-1, -1, -1]
    ]).astype(np.float64)

    spherical = cartesian_to_spherical(cartesian)

    pi_half = np.pi / 2

    expected = np.array([
        [1, 0, pi_half],
        [1, pi_half, pi_half],
        [1, 0, 0],
        [np.sqrt(3), np.pi / 4, np.arccos(1 / np.sqrt(3))],
        [np.sqrt(3), -np.deg2rad(135), np.arccos(-1 / np.sqrt(3))]
    ]).astype(np.float64)

    assert np.allclose(spherical, expected)

if __name__ == "__main__":
    points = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1]
    ]).astype(np.float64)
    kappa_max = 2
    points = np.random.uniform(-1, 1, size=(1000, 3))
    eta = np.array([0.001, 0.001, 0.001])
    coeffs = compress(points, 5)
    n = len(points)
    steps = 1000
    reconstruction = decompress(n, eta, steps, coeffs)
    import open3d as o3d
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(reconstruction)
    o3d.visualization.draw_geometries([pcd2])
