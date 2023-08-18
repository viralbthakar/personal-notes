"""
Unsupervised Point Cloud Segmentation with Gaussian Mixture Models

This program demonstrates unsupervised point cloud segmentation using Gaussian Mixture Models (GMM).
It generates synthetic point cloud data with optional Gaussian noise and visualizes the original and noisy point clouds.
Then, it performs a grid search to find the best parameters for the GMM model and applies the GMM for segmentation.

Example usage:
python main.py -s path_to_source_point_cloud

"""

import copy
import typing
import argparse
import numpy as np
import open3d as o3d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# Read point cloud files.
def read_point_clouds(file_path):
    """
    Read point cloud data from a file and return the open3d point cloud object, points, and normals.

    Parameters:
    ---------------
    file_path : str
        Path to the point cloud file.

    Returns:
    ---------------
    pcd : open3d.geometry.PointCloud
        Point cloud object read from the file.
    points : numpy.ndarray
        Array containing the x, y, z coordinates of the points in the point cloud.
    normals : numpy.ndarray
        Array containing the normal vectors of the points in the point cloud. If normals are not present in the
        original point cloud file, this function estimates normals using KDTreeSearchParamHybrid.

    Note:
    ---------------
    If the input point cloud does not have normals, this function estimates the normals using KDTreeSearchParamHybrid
    with default parameters (radius=0.5, max_nn=100).
    """
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    if normals.shape[0] == 0:
        print("Computing normals of points cloud")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100)
        )
        normals = np.asarray(pcd.normals)
    return pcd, points, normals


def points_to_open3d(points, normals=None, rgb=None):
    """
    Convert a set of points, optional normals, and optional RGB values into an Open3D point cloud object.

    Parameters:
    ---------------
    points : numpy.ndarray
        Array containing the coordinates of the points.
    normals : numpy.ndarray or None, optional
        Array containing the normal vectors of the points. If None, normals will be estimated using KDTreeSearchParamHybrid.
    rgb : numpy.ndarray or None, optional
        Array containing RGB color values for each point. If None, no colors will be assigned.

    Returns:
    ---------------
    pcd : open3d.geometry.PointCloud
        Point cloud object created from the input data.
    points_array : numpy.ndarray
        Array containing the coordinates of the points in the point cloud.
    normals_array : numpy.ndarray
        Array containing the normal vectors of the points in the point cloud.

    Note:
    ---------------
    If normals are not provided, this function estimates normals using KDTreeSearchParamHybrid with default parameters
    (radius=0.5, max_nn=100).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=100)
        )
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd, np.asarray(pcd.points), np.asarray(pcd.normals)


def visualize(
    points,
    normals=None,
    colors=None,
    uniform_color=True,
    mode="open3d",
    window_name="Open3D",
):
    """
    Visualize point cloud data using Open3D's visualization capabilities.

    Parameters:
    ---------------
    points : numpy.ndarray or list
        Array of point coordinates or list of arrays containing point coordinates for multiple point clouds.
    normals : numpy.ndarray or list or None, optional
        Array of normal vectors or list of arrays containing normal vectors for the points. If None, normals will be estimated.
    colors : numpy.ndarray or list or None, optional
        Array of RGB color values or list of arrays containing RGB color values for the points. If None, no colors will be assigned.
    mode : str, optional
        Visualization mode. Options: "open3d" (default).
    window_name : str, optional
        Name of the visualization window.

    Note:
    ---------------
    - If points is a list of arrays, each element in the list is treated as a separate point cloud.
    - If normals are not provided, they will be estimated using KDTreeSearchParamHybrid.
    - If colors are provided, they will be applied uniformly to the point cloud(s).
    - The function uses Open3D's visualization backend to display the point clouds.
    """
    if mode == "open3d":
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        if isinstance(points, list):
            for i, pts in enumerate(points):
                temp_pcd, _, _ = points_to_open3d(pts, normals=normals[i])
                if colors is not None:
                    if uniform_color:
                        temp_pcd.paint_uniform_color(colors[i])
                    else:
                        temp_pcd.colors = o3d.utility.Vector3dVector(colors[i])
                vis.add_geometry(temp_pcd)
        else:
            temp_pcd = points_to_open3d(points)
            if colors is not None:
                if uniform_color:
                    temp_pcd.paint_uniform_color(colors)
                else:
                    temp_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(temp_pcd)
        vis.run()


def add_gaussian_noise(point_cloud, mean=0, std_dev=0.01):
    """
    Add Gaussian noise to a given point cloud.

    Parameters:
    ---------------
    point_cloud : numpy.ndarray
        Array representing the point cloud data with shape (num_points, num_dimensions).
    mean : float, optional
        Mean of the Gaussian noise distribution (default is 0).
    std_dev : float, optional
        Standard deviation of the Gaussian noise distribution (default is 0.01).

    Returns:
    ---------------
    noisy_point_cloud : numpy.ndarray
        Array containing the original point cloud data with Gaussian noise added.

    Note:
    ---------------
    This function adds Gaussian noise to each point in the input point cloud, where the noise is generated from a
    Gaussian distribution with the specified mean and standard deviation. The noise is added element-wise to the
    coordinates of the points in the point cloud.
    """
    noise = np.random.normal(mean, std_dev, point_cloud.shape)
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud


def create_samples(point_cloud, normals, num_samples=10):
    """
    Create multiple samples by adding Gaussian noise to a given point cloud and its normals.

    Parameters:
    ---------------
    point_cloud : numpy.ndarray
        Array representing the original point cloud data with shape (num_points, num_dimensions).
    normals : numpy.ndarray
        Array representing the normal vectors of the original point cloud with shape (num_points, num_dimensions).
    num_samples : int, optional
        Number of samples to generate by adding Gaussian noise (default is 10).

    Returns:
    ---------------
    out_points : numpy.ndarray
        Array containing the original point cloud data and the generated noisy samples with shape
        (num_points * (num_samples + 1), num_dimensions).
    out_normals : numpy.ndarray
        Array containing the original normal vectors and the generated noisy normal vectors with shape
        (num_points * (num_samples + 1), num_dimensions).

    Note:
    ---------------
    This function creates multiple samples by adding Gaussian noise to the original point cloud and its corresponding
    normal vectors. The specified number of samples will be generated, each with independent noise. The original point
    cloud and normal vectors are included in the output along with the generated noisy samples.
    """
    original_point_cloud = point_cloud.copy()
    original_normals = normals.copy()

    out_points = point_cloud.copy()
    out_normals = normals.copy()

    for i in range(num_samples):
        noise_points = add_gaussian_noise(original_point_cloud)
        noise_pcd, noise_points, noise_normals = points_to_open3d(noise_points)
        out_points = np.vstack((out_points, noise_points))
        out_normals = np.vstack((out_normals, noise_normals))
    return out_points, out_normals


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsupervised Point Cloud Segmentation with GMM"
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default=None,
        help="File path to source point cloud. If None use dummy data.",
    )
    parser.add_argument(
        "-p",
        "--points",
        type=int,
        default=10,
        help="Number of synthetic points to add around each original point.",
    )
    parser.add_argument(
        "-n",
        "--use-normals",
        action="store_true",
        help="Flag to decide whether to use normals information in clustering or not.",
    )

    opt = parser.parse_args()
    np.random.seed(0)

    if opt.source is not None:
        source_pcd, source_points, source_normals = read_point_clouds(opt.source)
    else:
        source_points = np.vstack(
            (
                np.random.uniform(low=0.0, high=1.0, size=(1000, 3)),
                np.random.uniform(low=2.0, high=3.0, size=(1000, 3)),
            )
        )
        source_pcd, source_points, source_normals = points_to_open3d(source_points)

    print(f"The shape of input source point cloud is : {source_points.shape}")

    noise_points, noise_normals = create_samples(
        source_points, source_normals, opt.points
    )
    noise_pcd, noise_points, noise_normals = points_to_open3d(
        noise_points, noise_normals
    )

    print(
        f"The shape of input source point cloud after jittering : {noise_points.shape}"
    )

    visualize(
        points=[source_points, noise_points],
        normals=[source_normals, noise_normals],
        colors=[[0, 0, 1], [1, 1, 0]],
        mode="open3d",
        window_name="Input Source Point Cloud",
    )

    if opt.use_normals:
        data = np.hstack((noise_normals, noise_points))
    else:
        data = noise_normals.copy()

    print(f"The shape of input data for clustering is : {data.shape}")

    # Set up parameters for GridSearch
    param_grid = {
        "n_components": np.arange(2, 15),
        "covariance_type": ["full"],
    }
    print(f"Initiated Grid with parameters: {param_grid}")

    # Perform GridSearch with GMM
    gmm = GaussianMixture()
    grid_search = GridSearchCV(gmm, param_grid=param_grid)
    grid_search.fit(data)

    best_n_components = grid_search.best_params_["n_components"]
    best_covariance_type = grid_search.best_params_["covariance_type"]

    print(f"Found the best set of parameters: {grid_search.best_params_}")

    # Fit the GMM model with the best parameters
    best_gmm = GaussianMixture(
        n_components=best_n_components, covariance_type=best_covariance_type
    )
    best_gmm.fit(data)
    labels = best_gmm.predict(data)

    unique_labels = set(labels)
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0

    visualize(
        points=[source_points],
        normals=[source_normals],
        colors=[colors[:, :3]],
        uniform_color=False,
        mode="open3d",
        window_name="Clustered Point Cloud",
    )
