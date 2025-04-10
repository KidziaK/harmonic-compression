from enum import Enum
import quaternionic
import scipy
import spherical
from numba import njit, prange
from scipy.special import iv, lpmv
from tqdm import tqdm
from utility.e57 import load_e57, numpy_to_e57
import open3d as o3d
import numpy as np

l_max = 2
wigner = spherical.Wigner(l_max + 1) # We need +1 for recurrence relation for derivative of wigner d

def calculate_rmse_o3d(pcd1_np, pcd2_np):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_np)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_np)
    distances = pcd1.compute_point_cloud_distance(pcd2)
    rmse = np.sqrt(np.mean(np.asarray(distances)**2))
    return rmse

def calculate_chamfer_distance_o3d(pcd1_np, pcd2_np):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pcd1_np)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pcd2_np)
    dist_p1_to_p2 = pcd1.compute_point_cloud_distance(pcd2)
    dist_p2_to_p1 = pcd2.compute_point_cloud_distance(pcd1)
    chamfer_distance = np.mean(np.asarray(dist_p1_to_p2)**2) + np.mean(np.asarray(dist_p2_to_p1)**2)
    return chamfer_distance


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
    # return (B(l) * np.exp(kappa)).astype(np.complex128)

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

def calc_D(q):
    return wigner.D(q)

def compress(spherical_coords):
    kappa = spherical_coords[:, 0]
    theta = spherical_coords[:, 1]
    phi = spherical_coords[:, 2]

    l = np.arange(0, l_max + 1)

    l_grid, kappa_grid = np.meshgrid(l, kappa)

    f_tilda = g(l_grid, kappa_grid)

    q = quaternionic.array.from_euler_angles(np.zeros_like(theta), phi, theta)

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

def loss(spherical_coords_flat, true_coefficients):
    n = len(spherical_coords_flat)
    spherical_coords = spherical_coords_flat.reshape((3, n // 3)).T
    predicted_coefficients = compress(spherical_coords)
    F_tilda = predicted_coefficients.sum(axis=0)
    F = true_coefficients.sum(axis=0)
    diff = F_tilda - F
    L = np.real(np.conjugate(diff) * diff).sum()
    return L

def loss_theta_phi(theta_phi_flat, true_coefficients, kappa_start):
    n = len(theta_phi_flat)
    theta_phi = theta_phi_flat.reshape((2, n // 2)).T
    spherical_coords = np.dstack((kappa_start, theta_phi[:, 0], theta_phi[:, 1])).squeeze(0)
    predicted_coefficients = compress(spherical_coords)
    # F_tilda = predicted_coefficients.sum(axis=0)
    # F = true_coefficients.sum(axis=0)
    diff = predicted_coefficients - true_coefficients
    L = np.real(np.conjugate(diff) * diff).sum()
    return L

def loss_kappa(kappa, true_coefficients, theta, phi):
    spherical_coords = np.dstack((kappa, theta, phi)).squeeze(0)
    predicted_coefficients = compress(spherical_coords)
    # F_tilda = predicted_coefficients.sum(axis=0)
    # F = true_coefficients.sum(axis=0)
    diff = predicted_coefficients - true_coefficients
    L = np.real(np.conjugate(diff) * diff).sum()
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



class PointCloud(Enum):
    ST_SULPICE = "StSulpice-Cloud-50mm"
    CUBE = "cube"
    BUNNY = "Bunny"

if __name__ == "__main__":
    debug_info = False
    save_reconstruction = True
    point_cloud_name = PointCloud.ST_SULPICE.value

    point_cloud, colors = load_e57(f"../data/{point_cloud_name}.e57")
    reconstructed_points = np.zeros_like(point_cloud)
    radius_max = np.linalg.norm(point_cloud, axis=1, keepdims=True).max().astype(np.float32)

    batch_size = n = 32
    for i in tqdm(range(len(point_cloud) // batch_size)):
    # for i in range(10):
        left = i * batch_size
        right = min(len(point_cloud), (i + 1) * batch_size)
        points = point_cloud[left:right]

        points = points / radius_max
        spherical_coords = cartesian_to_spherical(points)

        kappa = spherical_coords[:, 0].astype(np.float64)
        theta = spherical_coords[:, 1].astype(np.float64)
        phi = spherical_coords[:, 2].astype(np.float64)

        kappa_start = 0.5 * np.ones_like(spherical_coords[:, 0])
        theta_start = np.linspace(0, 2 * np.pi, len(theta))
        phi_start = np.linspace(0, np.pi, len(phi))

        harmonic_coefficients = compress(spherical_coords).astype(np.complex64)

        bounds_kappa = np.array([[0, 1] for _ in range(n)])

        reconstruction_theta_phi = scipy.optimize.minimize(
            loss_theta_phi,
            np.hstack([theta_start, phi_start]),
            (harmonic_coefficients, kappa_start),
            # bounds=np.concatenate([bounds_theta, bounds_phi]),
            # jac=grad_loss,
            options={"disp": debug_info}
        )

        theta_reconstructed = reconstruction_theta_phi.x[:n]
        phi_reconstructed = reconstruction_theta_phi.x[n:2 * n]

        reconstruction_kappa = scipy.optimize.minimize(
            loss_kappa,
            kappa_start,
            (harmonic_coefficients, theta_reconstructed, phi_reconstructed),
            bounds=bounds_kappa,
            # jac=grad_loss,
            options={"disp": debug_info}
        )

        kappa_reconstructed = reconstruction_kappa.x[:n]

        reconstruction_theta_phi = scipy.optimize.minimize(
            loss_theta_phi,
            np.hstack([theta_reconstructed, phi_reconstructed]),
            (harmonic_coefficients, kappa_reconstructed),
            # bounds=np.concatenate([bounds_theta, bounds_phi]),
            # jac=grad_loss,
            options={"disp": debug_info}
        )

        theta_reconstructed = reconstruction_theta_phi.x[:n]
        phi_reconstructed = reconstruction_theta_phi.x[n:2 * n]

        reconstruction_kappa = scipy.optimize.minimize(
            loss_kappa,
            kappa_reconstructed,
            (harmonic_coefficients, theta_reconstructed, phi_reconstructed),
            bounds=bounds_kappa,
            # jac=grad_loss,
            options={"disp": debug_info}
        )

        kappa_reconstructed = reconstruction_kappa.x[:n]

        reconstructed_points[left:right] = spherical_to_cartesian(kappa_reconstructed, theta_reconstructed, phi_reconstructed) * radius_max

    rmse_o3d = calculate_rmse_o3d(reconstructed_points, point_cloud)
    chamfer_o3d = calculate_chamfer_distance_o3d(reconstructed_points, point_cloud)

    print(f"RMSE (Open3D): {rmse_o3d}")
    print(f"Chamfer Distance (Open3D): {chamfer_o3d}")

    if save_reconstruction:
        if colors is None:
            colors = (255 * np.ones(shape=(len(reconstructed_points), 3))).astype(np.uint8)

        numpy_to_e57(f"{point_cloud_name}-reconstructed-{batch_size}-v2.e57", reconstructed_points, colors)
