import numpy as np

# from numba import njit
# from scipy.special import iv, lpmv, factorial
# from typing import Tuple
# from functools import cache

from compression.math import *

class Compressor:
    def __init__(self, l: int) -> None:
        self.l = l

    def __call__(self, points: np.ndarray) -> np.ndarray:
        spherical = cartesian_to_spherical(points)

        kappa = spherical[:, 0]
        theta = spherical[:, 1]
        phi = spherical[:, 2]

        l = np.arange(self.l)

        coefficients = np.zeros(shape=(self.l, self.l))

        # for l in range(l_max):
        #     for m in range(l+1):
        #         coefficients[(m, l)] = harmonic_f(m, l, kappa, theta, phi)

        f_tilda = standard_mises_fisher_harmonic_coefficients(l, kappa)
        d_wigner = wigner_d_function(l, l, theta, phi)

        f = f_tilda[:, np.newaxis, :] * d_wigner

        print(reconstruct_from_harmonics(f, np.linspace(0, 2 * np.pi, num=10), np.linspace(0, np.pi, num=10)))

        return coefficients

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

c = Compressor(8)
c(points)

# @cache
# def B(l: int):
#     return np.sqrt((2 * l + 1) / 4 / np.pi)

# def harmonic_tilda(l: int, kappa: np.ndarray) -> np.ndarray:
#     """ 
#     Harmonic coefficient of order l of standard (mu is z-axis aligned) 3D von-mises-fisher distribution.
#     Since mu = (0, 0, 1), all complex spherical harmonic coefficients are equal to 0.
#     """
#     return B(l) * iv(l + 0.5, kappa) / iv(0.5, kappa)



# def harmonic_f(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return np.sum(harmonic_tilda(l, kappa) * wigner_d_function(m, l, theta, phi))

# def compress(point_cloud: np.ndarray, l_max: int) -> dict:
#     spherical = cartesian_to_spherical(point_cloud)
#     coefficients = {}

#     kappa = spherical[:, 0]
#     theta = spherical[:, 1]
#     phi = spherical[:, 2]

#     for l in range(l_max):
#         for m in range(l+1):
#             coefficients[(m, l)] = harmonic_f(m, l, kappa, theta, phi)

#     return coefficients

# def d_d_matrix_d_phi(m: int, l: int, phi: np.ndarray) -> np.ndarray:
#     cos_phi = np.cos(phi)
#     sin_phi = np.sin(phi) + 1e-5
#     return A(m, l) * (1 / sin_phi) * (lpmv(m + 1, l, cos_phi) + m * cos_phi * lpmv(m, l, cos_phi))

# def d_func_d_phi(m: int, l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return np.exp(-1.0j * theta * m) * d_d_matrix_d_phi(m, l, phi)

# def d_func_d_theta(m: int, l: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return -1.0j * m * wigner_d_function(l, m, theta, phi)

# @cache
# def N(m: int, l: int):
#     return A(m, l) * B(l)

# def c(kappa: np.ndarray) -> np.ndarray:
#     return np.sqrt(kappa) / np.power(2 * np.pi, 3/2) * iv(0.5, kappa)

# @njit
# def d_c_d_kappa(kappa: np.ndarray) -> np.ndarray:
#     return (np.sinh(kappa) - kappa * np.cosh(kappa)) / np.pi / 4 / np.sinh(kappa) / np.sinh(kappa)

# @njit
# def IK(kappa: np.ndarray) -> np.ndarray:
#     return np.sqrt(2 * np.pi / kappa) 

# @njit
# def d_IK_d_kappa(kappa: np.ndarray) -> np.ndarray:
#     return -np.sqrt(2 * np.pi) * np.power(kappa, 1.5) / 2

# def I(kappa: np.ndarray, l: int) -> np.ndarray:
#     return IK(kappa) * iv(l + 0.5, kappa)

# def d_iv_d_kappa(kappa: np.ndarray, l: int) -> np.ndarray:
#     return (iv(l - 0.5, kappa) + iv(l + 1.5, kappa)) / 2

# def d_I_d_kappa(kappa: np.ndarray, l: int) -> np.ndarray:
#     return d_IK_d_kappa(kappa) * iv(l + 0.5, kappa) + IK(kappa) * d_iv_d_kappa(kappa, l)

# def d_g_tilda_d_kappa(l: int, m: int, kappa: np.ndarray) -> np.ndarray:
#     return 2 * np.pi * N(m, l) * (d_c_d_kappa(kappa) * I(kappa, l) / c(kappa) + d_I_d_kappa(kappa, l) * c(kappa))

# def d_g_d_kappa(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return wigner_d_function(l, m, theta, phi) * d_g_tilda_d_kappa(l, m, kappa)

# def d_g_d_theta(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return d_func_d_theta(m, l, theta, phi) * harmonic_tilda(l, kappa)

# def d_g_d_phi(m: int, l: int, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
#     return d_func_d_phi(m, l, theta, phi) * harmonic_tilda(l, kappa)

# def L(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
#     loss = 0
#     for (m, l), f in coeffs.items():
#         g = harmonic_f(m, l, kappa, theta, phi)
#         inner = np.abs(np.sum(g) - f)
#         loss += inner * inner
#     return loss

# def d_L_d_theta(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
#     d_loss = 0
#     for (m, l), f in coeffs.items():
#         g = harmonic_f(m, l, kappa, theta, phi)
#         sum_g = np.sum(g)
#         dg = d_g_d_theta(m, l, kappa, theta, phi)
#         inner_diff = sum_g - f
#         d_loss += np.real(dg * np.conjugate(inner_diff) + np.conjugate(dg) * inner_diff)
#     return d_loss

# def d_L_d_phi(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
#     d_loss = 0
#     for (m, l), f in coeffs.items():
#         g = harmonic_f(m, l, kappa, theta, phi)
#         sum_g = np.sum(g)
#         dg = d_g_d_phi(m, l, kappa, theta, phi)
#         inner_diff = sum_g - f
#         d_loss += np.real(dg * np.conjugate(inner_diff) + np.conjugate(dg) * inner_diff)
#     return d_loss

# def d_L_d_kappa(coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> float:
#     d_loss = 0
#     for (m, l), f in coeffs.items():
#         g = harmonic_f(m, l, kappa, theta, phi)
#         sum_g = np.sum(g)
#         dg = d_g_d_kappa(m, l, kappa, theta, phi)
#         inner_diff = sum_g - f
#         d_loss += np.real(dg * np.conjugate(inner_diff) + np.conjugate(dg) * inner_diff)
#     return d_loss

# def grad_step(eta: np.ndarray, coeffs: dict, kappa: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     dk, dt, dp = d_L_d_kappa(coeffs, kappa, theta, phi), d_L_d_theta(coeffs, kappa, theta, phi), d_L_d_phi(coeffs, kappa, theta, phi)
#     # dk = dt / dt.max()
#     # dk = dp / dp.max()
#     # kappa = np.clip(kappa - eta[0] * dk, 1e-5, 10)
#     theta = np.mod(theta - eta[1] * dt, 2 * np.pi)
#     phi = np.mod(phi - eta[2] * dp, np.pi)
#     print(L(coeffs, kappa, theta, phi), kappa.min(), theta.min(), phi.min())
#     # print(np.sum(dk < 0))
#     # print(dk.max(), dt.max(),dp.max())

#     return kappa, theta, phi

# def decompress(n: int, eta: np.ndarray, steps: int, coeffs: dict) -> np.ndarray:
#     uniform_points = np.random.standard_normal(size=(n, 3))
#     uniform_points = uniform_points / np.linalg.norm(uniform_points, axis=1)[:, np.newaxis]
#     spherical = cartesian_to_spherical(uniform_points)

#     kappa = spherical[:, 0] * np.sqrt(3)
#     theta = spherical[:, 1]
#     phi = spherical[:, 2]

#     for i in range(steps):
#         kappa, theta, phi = grad_step(eta, coeffs, kappa, theta, phi)
        

#     return spherical_to_cartesian(kappa, theta, phi)

# if __name__ == "__main__":
#     points = np.array([
#         [-1, -1, -1],
#         [1, -1, -1],
#         [-1, 1, -1],
#         [-1, -1, 1],
#         [1, 1, -1],
#         [1, -1, 1],
#         [-1, 1, 1],
#         [1, 1, 1]
#     ]).astype(np.float64)
#     kappa_max = 2
#     # points = np.random.uniform(-1, 1, size=(1000, 3))
#     eta = np.array([0.001, 0.001, 0.001])
#     coeffs = compress(points, 5)
#     n = len(points)
#     steps = 100000
#     reconstruction = decompress(n, eta, steps, coeffs)
#     import open3d as o3d
#     # pcd1 = o3d.geometry.PointCloud()
#     # pcd1.points = o3d.utility.Vector3dVector(points)
    
#     pcd2 = o3d.geometry.PointCloud()
#     pcd2.points = o3d.utility.Vector3dVector(reconstruction)
#     o3d.visualization.draw_geometries([pcd2])

    
