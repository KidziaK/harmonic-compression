import e57
import open3d as o3d
import numpy as np
import scipy

from enum import Enum, auto
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from typing import Callable
from math_utils import *

class PointCloudVisualizer():

    def __init__(self, cloud_path : str):
        pc = e57.read_points(cloud_path)
        self._point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.points))
        self._point_cloud.colors = o3d.utility.Vector3dVector(pc.color.astype('float64'))

    def visualize(self):
        o3d.visualization.draw_geometries([self._point_cloud])

class SphereSignalVisualzier:
    class VisualizerMode(Enum):
        HEAT_MAP = auto()

    def __init__(self, mode: VisualizerMode = VisualizerMode.HEAT_MAP, resolution: int = 50):
        """
        Args:
            mode: Visualization method by deafult show spherical signal as a heat map on a sphere.
            resolution: Number of uniform samples on a unit sphere.
        """
        self.mode = mode
        self.resolution = resolution
        
    def visualize(self, signal: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        match self.mode:
            case self.VisualizerMode.HEAT_MAP:
                self._visualize_heat_map(signal)
            case _:
                raise NotImplementedError("Only avaialble VisualizationMode is HEAT_MAP")

    def _visualize_heat_map(self, signal: Callable[[np.ndarray, np.ndarray], None]):
        theta = 2 * np.pi * np.random.uniform(size=self.resolution)
        phi =   np.arccos(2 * np.random.uniform(size=self.resolution) - 1)
        r = signal(theta, phi)
        
        xyz = spherical_to_cartesian(np.ones_like(theta), theta, phi)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=r, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
if __name__ == "__main__":
    visualzier = SphereSignalVisualzier(resolution=10000)

    def signal(theta, phi):
        vmf = scipy.stats.vonmises_fisher([1, 0, 0], 0.1)
        return vmf.pdf(spherical_to_cartesian(np.ones_like(theta), theta, phi))

    visualzier.visualize(signal)
