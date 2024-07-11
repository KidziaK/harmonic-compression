import e57
import open3d as o3d

class PointCloudVisualizer():

    def __init__(self, cloud_path : str):
        pc = e57.read_points(cloud_path)
        self._point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc.points))
        self._point_cloud.colors = o3d.utility.Vector3dVector(pc.color.astype('float64'))

    def visualize(self):
        o3d.visualization.draw_geometries([self._point_cloud])