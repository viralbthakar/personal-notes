# Iterative Closest Point Algorithm

## Description
The Iterative Closest Point (ICP) algorithm is a widely used method in computer vision and robotics to align two sets of points in a 3D space. It is often employed to find the transformation (translation and rotation) that best aligns one point cloud with another.

Imagine you have two sets of 3D points: `a source point cloud` and `a target point cloud`. 

```The goal of the ICP algorithm is to find the best rigid transformation (translation and rotation) that minimizes the distance between the points in the source cloud and their corresponding points in the target cloud. These corresponding points are called "closest points."```

Here's a step-by-step explanation of the ICP algorithm for beginners:

*1. Initialization:*

Start with an initial guess for the transformation (translation and rotation) between the source and target point clouds. This could be a rough estimation or simply the identity transformation (no transformation at all).

*2. Correspondence:*

For each point in the source point cloud, find its nearest neighbor in the target point cloud based on the Euclidean distance. These nearest neighbors are the "corresponding points."

*3. Error Calculation:*

Calculate the error between each point in the source cloud and its corresponding point in the target cloud. The error is typically represented as the sum of squared distances between corresponding points.

*4. Weighting (Optional):*

Optionally, you can assign weights to the points based on their reliability or importance. For example, points with higher accuracy could be given higher weights, and points with lower accuracy could be down-weighted.

*5. Transformation Estimation:*

Using the corresponding points and optionally their weights, estimate the transformation that best aligns the source point cloud with the target point cloud. The most common transformation is a combination of translation and rotation.

*6. Apply Transformation:*

Apply the estimated transformation to the source point cloud, so it moves closer to the target point cloud.

*7. Termination Condition:*

Check if the error is below a certain threshold or if the transformation parameters have converged. If not, go back to step 2 and repeat the process with the updated source point cloud.

*8. Final Transformation:*

Once the error is below the threshold or the transformation converges, you have found the best transformation to align the two point clouds.

*9. Output:*
The final transformation matrix, which describes the translation and rotation needed to align the source point cloud with the target point cloud.

ICP is an iterative process, and it keeps refining the transformation until it reaches an acceptable alignment. This algorithm is widely used in various applications, such as point cloud registration, object recognition, and 3D reconstruction. However, it's worth noting that ICP might not always converge to the correct solution, especially when dealing with noisy or incomplete data. In such cases, other variants or extensions of ICP can be used to improve the alignment accuracy.

## Reading Resources
1. ICP Registration - Open3D Example - [URL](http://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html#ICP-registration)
2. Affine Transforamtions - [URL](https://people.computing.clemson.edu/~dhouse/courses/401/notes/affines-matrices.pdf)
3. Paul J. Besl and Neil D. McKay, A Method for Registration of 3D Shapes, PAMI, 1992. - [PDF](https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_icp.pdf)
4. Y.Chen and G. G. Medioni, Object modelling by registration of multiple range images, Image and Vision Computing, 10(3), 1992. - [PDF](https://graphics.stanford.edu/courses/cs348a-17-winter/Handouts/chen-medioni-align-rob91.pdf)
5. J.Park, Q.-Y. Zhou, and V. Koltun, Colored Point Cloud Registration Revisited, ICCV, 2017. - [PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Park_Colored_Point_Cloud_ICCV_2017_paper.pdf)