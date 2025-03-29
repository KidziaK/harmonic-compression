import pye57 as e57
import cv2

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from pathlib import Path
from numpy.typing import NDArray

from utility.e57 import load_e57
from visualizer import SphericalMapVisualizer


POINT_CLOUD_PATH = Path("./data/StSulpice-Cloud-50mm.e57")
TEXTURE_RESOLUTION = (1024, 1024)

@nb.njit(parallel=True)
def von_mises_fisher_3d(theta: NDArray[np.float64], phi: NDArray[np.float64], mu: NDArray[np.float64], kappa: np.float64) -> NDArray[np.float64]:
    C3 = kappa / (2 * np.pi * (np.exp(kappa) - np.exp(-kappa)))
    mu_dot_x = mu[0] * np.sin(phi) * np.cos(theta) + mu[1] * np.sin(phi) * np.sin(theta) + mu[2] * np.cos(phi)
    exp_kappa_theta_mu = np.exp(kappa * mu_dot_x)
    return C3 * exp_kappa_theta_mu

@nb.njit(parallel=True)
def create_heat_map(points: NDArray[np.float64], theta: NDArray[np.float64], phi: NDArray[np.float64]) -> NDArray[np.float64]:
    heat_map = np.zeros_like(theta)

    n = len(points)

    for i in nb.prange(n):
        kappa = np.linalg.norm(points[i])
        mu = points[i] / kappa
        heat_map += von_mises_fisher_3d(theta, phi, mu, kappa) / n

    return heat_map

if __name__ == "__main__":
    height, width = TEXTURE_RESOLUTION
    point_cloud_cached_name = f"{POINT_CLOUD_PATH.stem}_{height}_{width}.hdr"
    point_cloud_cache_dir = POINT_CLOUD_PATH.parents[1].resolve().joinpath("cache")
    point_cloud_cache_dir.mkdir(exist_ok=True)
    point_cloud_cached_path = point_cloud_cache_dir.joinpath(point_cloud_cached_name)

    if point_cloud_cached_path.exists():
        heat_map = cv2.imread(str(point_cloud_cached_path), cv2.IMREAD_UNCHANGED)
    else:
        points = load_e57(POINT_CLOUD_PATH)

        radius = np.linalg.norm(points, axis=1, keepdims=True)
        radius_scaled = (np.arctan(radius) / np.pi + 0.5) * 40 + 10

        points = points / radius * radius_scaled

        theta = np.linspace(0, 2 * np.pi, width)
        phi = np.linspace(0, np.pi, height)

        theta, phi = np.meshgrid(theta, phi)
        heat_map = create_heat_map(points, theta, phi)
        heat_map = plt.cm.jet(heat_map)[:, :, :3]
        heat_map = heat_map.astype(np.float32)
        cv2.imwrite(str(point_cloud_cached_path), heat_map)
    
    v = SphericalMapVisualizer(heat_map)
    v.run()
