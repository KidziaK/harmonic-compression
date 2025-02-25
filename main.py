import e57
import cv2

import numpy as np
import numba as nb

from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from visualizer import SphericalMapVisualizer
from compression.math import cartesian_to_spherical, spherical_to_cartesian


POINT_CLOUD_PATH = Path("./data/Bunny.e57")
TEXTURE_RESOLUTION = (1024, 1024)

@nb.njit(parallel=True, cache=True)
def von_mises_fisher_3d(theta: float, phi: float, mu: NDArray[np.float32], kappa: float) -> float:
    C3 = kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))
    mu_dot_x = mu[0] * np.sin(phi) * np.cos(theta) + mu[1] * np.sin(phi) * np.sin(theta) + mu[2] * np.cos(phi)
    exp_kappa_theta_mu = np.exp(kappa * mu_dot_x)
    return C3 * exp_kappa_theta_mu

@nb.njit(parallel=True, cache=True)
def create_heat_map(points: NDArray[np.float32], theta: NDArray[np.float32], phi: NDArray[np.float32]) -> NDArray[np.float32]:
    heat_map = np.zeros_like(theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    n = len(points)

    for i in nb.prange(n):
        kappa = np.linalg.norm(points[i])
        mu = points[i] / kappa

        weight = 1 / (i + 1)

        heat_map = (1 - weight) * heat_map + weight * von_mises_fisher_3d(theta, phi, mu, kappa)

    return heat_map

if __name__ == "__main__":
    height, width = TEXTURE_RESOLUTION
    point_cloud_cached_name = f"{POINT_CLOUD_PATH.stem}_{height}_{width}.hdr"
    point_cloud_cache_dir = POINT_CLOUD_PATH.parents[1].resolve().joinpath("cache")
    point_cloud_cached_path = point_cloud_cache_dir.joinpath(point_cloud_cached_name)

    if point_cloud_cached_path.exists():
        heat_map = cv2.imread(str(point_cloud_cached_path), cv2.IMREAD_UNCHANGED)
    else:
        point_cloud = e57.read_points(str(POINT_CLOUD_PATH)).points
        point_cloud_spherical = cartesian_to_spherical(point_cloud)
        
        theta = np.linspace(0, 2 * np.pi, width)
        phi = np.linspace(0, np.pi, height)

        theta, phi = np.meshgrid(theta, phi)
        heat_map = create_heat_map(point_cloud, theta, phi)
        heat_map = cv2.applyColorMap(heat_map, cv2.COLORMAP_JET)
        cv2.imwrite(str(point_cloud_cached_path), heat_map)
    
    # v = SphericalMapVisualizer(heat_map)
    # v.run()
