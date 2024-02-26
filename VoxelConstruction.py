import cv2
import numpy as np

def load_camera_parameters(camera_config_path):
    # OpenCV's cv2.FileStorage for reading might have limitations in Python.
    # Consider using an XML parser for robustness.
    fs = cv2.FileStorage(camera_config_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    rvec = fs.getNode("RotationVectors").mat()
    tvec = fs.getNode("TranslationVectors").mat()
    fs.release()
    return camera_matrix, dist_coeffs, rvec, tvec

def create_lookup_table(voxel_volume_bounds, resolution, camera_configs):
    xv, yv, zv = np.mgrid[
        voxel_volume_bounds[0][0]:voxel_volume_bounds[0][1]:resolution,
        voxel_volume_bounds[1][0]:voxel_volume_bounds[1][1]:resolution,
        voxel_volume_bounds[2][0]:voxel_volume_bounds[2][1]:resolution]
    
    voxel_points = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).transpose().astype(np.float32)
    lookup_table = []

    for c, config_path in enumerate(camera_configs, start=1):
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(config_path)

        # Project voxel points to the image plane for this camera
        img_points, _ = cv2.projectPoints(voxel_points, rvec, tvec, camera_matrix, dist_coeffs)

        for voxel, img_point in zip(voxel_points, img_points):
            xim, yim = img_point.ravel()  # Extract projected x, y coordinates
            # Project points returns the xim yim as floats so we round them to integers
            xim_int = int(round(xim))
            yim_int = int(round(yim))
            lookup_table.append((*voxel, c, xim_int, yim_int))  # Append voxel coord, camera index, and img coord

    return lookup_table

def load_foreground_masks():
    masks = {}
    for cam_id in range(1, 5):  # Assuming you have 4 cameras
        mask_path = f"data/cam{cam_id}/voxel_construction_frame.jpg"
        masks[cam_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return masks

def is_voxel_visible_in_camera(voxel_xim, voxel_yim, mask):
    # Check if the coordinates are within the mask bounds
    if 0 <= voxel_yim < mask.shape[0] and 0 <= voxel_xim < mask.shape[1]:
        # Return True if the pixel is foreground (white)
        # Can just do == 255 but this is more robust
        return mask[voxel_yim, voxel_xim] > 128
    return False

# def perform_voxel_reconstruction(lookup_table, masks):
#     reconstructed_voxels = []

#     for voxel in lookup_table:
#         voxel_xim, voxel_yim = voxel[4:]  # Assuming the first 3 elements are the voxel coordinates
#         visibility_count = 0
        
#         # Cam id and mask as a dictionary in case we want to use more than 1 frame of each camera in the future
#         for cam_id, mask in masks.items():
#             if is_voxel_visible_in_camera(voxel_xim, voxel_yim, mask):
#                 visibility_count += 1

#         # Mark the voxel as "on" if visible in at least 3 out of 4 cameras
#         if visibility_count >= 3:
#             reconstructed_voxels.append(voxel[:3])

#     return reconstructed_voxels

def main():
    
    world_width = 32
    world_height = 8
    world_depth = 32
    block_size = 1.0
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]

    # Create a voxel grid with a resolution of 1
    voxel_volume_bounds = [
    (-world_width/2 * block_size, world_width/2 * block_size),  # X-axis bounds
    (-block_size, world_height * block_size),  # Y-axis bounds
    (-world_depth/2 * block_size, world_depth/2 * block_size)  # Z-axis bounds
]
    resolution = 1

    voxel_lookup_table = create_lookup_table(voxel_volume_bounds, resolution, camera_configs)

    masks = load_foreground_masks()

    print(len(voxel_lookup_table))
    print(masks[1].shape)
    print([item[-2:] for item in voxel_lookup_table[0:10]])

    points_of_interest = [(10, 10), (20, 20), (300, 30)]  # Example points

    for point in points_of_interest:
        x, y = point
        print(f"Value at ({x}, {y}):", masks[1][y, x])

    non_zero_pixels = np.nonzero(masks[1])
    print('Non-zero pixel coordinates:', non_zero_pixels)
    print('Number of non-zero pixels:', len(non_zero_pixels[0])) 

    # reconstructed_voxels = perform_voxel_reconstruction(voxel_lookup_table, masks)

    

if __name__ == "__main__":
    main()