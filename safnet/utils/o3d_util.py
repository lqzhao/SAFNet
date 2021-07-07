"""o3d visualization helpers"""
import numpy as np
import open3d as o3d
import pdb


def draw_point_cloud(points, colors=None, normals=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def visualize_point_cloud(points, colors=None, normals=None, show_frame=False):
    pc = draw_point_cloud(points, colors, normals)
    geometries = [pc]
    if show_frame:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))
    pdb.set_trace()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(geometries)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(geometries)
    vis.destroy_window()

    o3d.visualization.draw_geometries(geometries)
