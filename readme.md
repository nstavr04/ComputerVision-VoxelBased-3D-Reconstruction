# Assignment 2: Voxel-based 3D Reconstruction

This assignment involves calibrating four cameras, performing background subtraction for each view, and implementing the silhouette-based voxel reconstruction algorithm.

## Tasks

1. **Calibration**: Obtain the intrinsic camera parameters, for every camera. Use the `intrinsics.avi` files of each view. Use the `checkerboard.avi` files to calculate the camera extrinsics.

2. **Background subtraction**: Use the `background.avi` files to create a background model. Once a background model is made, we will be doing background subtraction on each frame of `video.avi`.

3. **Voxel reconstruction**: Develop the voxel reconstruction algorithm such that the voxels that correspond to the person in the scene are "on", based on the four foreground images.