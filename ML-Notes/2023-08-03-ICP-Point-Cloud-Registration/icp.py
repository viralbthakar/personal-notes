"""
Iterative Closest Point (ICP) Algorithm with Open3d and NumPy.

This module provides an demostration of the ICP algorithm to align two sets of 3D points.
The algorithm iteratively estimates the rigid transformation (translation and rotation)
that minimizes the distance between the source and target point clouds.

Functions:
- read_point_clouds(file_path, to_numpy=False)
- points_to_open3d(points)
- perform_icp(source_points, target_points, max_correspondence_distance, init, max_iterations)
- visualize(points, colors=None, mode="open3d", window_name="Open3D")
- evaluate(source_points, target_points, max_correspondence_distance, transformation, mode="open3d")

Example usage:
- Run the script with command line arguments to specify source and target point cloud files, maximum correspondence distance, and maximum iterations.
- If no source or target file is specified, random point clouds will be generated.
- The script will output the initial alignment evaluation, perform ICP alignment, and output the evaluation after alignment.
- The script will also visualize the input and output point clouds.
"""
import copy
import typing
import argparse
import numpy as np
import open3d as o3d


# Read point cloud files.
def read_point_clouds(file_path, to_numpy=False):
    pld = o3d.io.read_point_cloud(file_path)
    if to_numpy:
        pld = np.asarray(pld.points)
    return pld


def points_to_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def perform_icp(
    source_points,
    target_points,
    max_correspondence_distance,
    init,
    max_iterations,
):
    source = points_to_open3d(source_points)
    target = points_to_open3d(target_points)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source,
        target,
        max_correspondence_distance=max_correspondence_distance,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )
    source.transform(reg_p2p.transformation)
    return reg_p2p, reg_p2p.transformation, np.asarray(source.points)


def visualize(points, colors=None, mode="open3d", window_name="Open3D"):
    if mode == "open3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        if isinstance(points, list):
            for i, pts in enumerate(points):
                temp_pcd = points_to_open3d(pts)
                if colors is not None:
                    temp_pcd.paint_uniform_color(colors[i])
                vis.add_geometry(temp_pcd)
        else:
            temp_pcd = points_to_open3d(points)
            if colors is not None:
                temp_pcd.paint_uniform_color(colors)
            vis.add_geometry(temp_pcd)
        vis.run()


def evaluate(
    source_points,
    target_points,
    max_correspondence_distance,
    transformation,
    mode="open3d",
):
    source = points_to_open3d(source_points)
    target = points_to_open3d(target_points)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, max_correspondence_distance, transformation
    )
    return evaluation


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Iterative Closest Point (ICP) Algorithm"
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=None,
        help="File path to source point cloud. If None use dummy data.",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="File path to target point cloud. If None use dummy data.",
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        default=0.5,
        help="Maximum correspondence points-pair distance",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1000,
        help="Maximum iteration before iteration stops",
    )

    opt = parser.parse_args()
    np.random.seed(0)

    if opt.source is not None:
        source_points = read_point_clouds(opt.source, to_numpy=True)
    else:
        source_points = np.random.rand(1000, 3)

    if opt.target is not None:
        target_points = read_point_clouds(opt.target, to_numpy=True)
    else:
        translation = np.random.uniform(-1, 1, size=3)
        rotation_angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        print(f"Translation {translation} and Rotation {rotation_angle}")

        target_points = np.dot(source_points, rotation_matrix.T) + translation

    # Identity transformation matrix for initialization.
    # No initial rotation or translation applied to target.
    init_transformation = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    print("Evaluate Initial Alignment")
    print(
        evaluate(
            source_points,
            target_points,
            max_correspondence_distance=opt.threshold,
            transformation=init_transformation,
        )
    )
    visualize(
        points=[source_points, target_points],
        colors=[[0, 0, 1], [0, 1, 0]],
        mode="open3d",
        window_name="Input Source - Target Point Cloud",
    )

    reg_p2p, transformation, transformed_source = perform_icp(
        source_points=source_points,
        target_points=target_points,
        max_correspondence_distance=opt.threshold,
        init=init_transformation,
        max_iterations=opt.iterations,
    )
    print("Evaluate After ICP Alignment")
    print(
        evaluate(
            source_points,
            target_points,
            max_correspondence_distance=opt.threshold,
            transformation=transformation,
        )
    )

    visualize(
        points=[source_points, target_points, transformed_source],
        colors=[[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        mode="open3d",
        window_name="Output Source - Target Point Cloud",
    )
