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
            lookup_table.append((*voxel, c, xim_int, yim_int))

    return lookup_table

def load_foreground_masks():
    masks = {}
    for cam_id in range(1, 5):  # Assuming you have 4 cameras
        mask_path = f"data/cam{cam_id}/voxel_construction_frame.jpg"
        masks[cam_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return masks

def is_voxel_visible_in_camera(voxel_xim, voxel_yim, mask):
    # Check if the coordinates are within the mask bounds
    if 0 <= int(voxel_yim) < mask.shape[0] and 0 <= int(voxel_xim) < mask.shape[1]:
        # Return True if the pixel is foreground (white)
        return mask[int(voxel_yim), int(voxel_xim)] == 255
    return False

# Search by voxel
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
#         if visibility_count >= 2:
#             reconstructed_voxels.append(voxel[:3])

#     return reconstructed_voxels

# Search by camera
def perform_voxel_reconstruction(lookup_table, masks):
    # Initialize a dictionary to hold the visibility status of each voxel
    # Keys are voxel coordinates as a tuple (x, y, z), values are counts of how many views the voxel is visible in
    voxel_visibility = {}

    # Iterate over every camera view c
    for cam_id, mask in masks.items():
        # Iterate through each entry in the lookup table
        for entry in lookup_table:
            # Extract voxel information and the camera view from the lookup table entry
            voxel_coords = tuple(entry[0]) # Extract voxel coordinates (x, y, z)
            voxel_cam_id = entry[1]          # Camera ID for this voxel entry
            voxel_xim, voxel_yim = entry[2] # Corresponding pixel coordinates in the camera view

            # Check if the current entry corresponds to the current camera view
            if voxel_cam_id == cam_id:
                # Check if the corresponding pixel is part of the foreground in this camera's view
                if is_voxel_visible_in_camera(voxel_xim, voxel_yim, mask):
                    # If so, mark the voxel as visible for this view
                    if voxel_coords not in voxel_visibility:
                        voxel_visibility[voxel_coords] = 1  # Mark as visible for the first time
                    else:
                        voxel_visibility[voxel_coords] += 1 # Increment visibility count

    # Reconstruct the list of voxels visible in at least one camera view
    reconstructed_voxels = [coords for coords, visibility_count in voxel_visibility.items() if visibility_count > 3]

    return reconstructed_voxels

def newmethod_lookup(width, height, depth, block_size):
    # x, y, z = np.meshgrid(np.arange(0, width, block_size), 
    #                       np.arange(0, height, block_size), 
    #                       np.arange(0, depth, block_size), indexing='ij')

    # # Reshape the coordinate grids to lists of coordinates
    # # Its a 3D grid so we need to flatten it to a 2D grid where a row is (number of voxel 0 - 1048576, 3)
    # voxel_volume = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T 

    x_range = np.linspace(-600, 900, num=100)
    y_range = np.linspace(-800, 800, num=100)
    z_range = np.linspace(-2000, 600, num=100)
    voxel_volume = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    print(voxel_volume.shape)

    # Convert to float32 for compatibility with cv2.projectPoints
    #voxel_3d = np.array(voxel_volume, dtype=np.float32)

    lookup_table = []

    #print(voxel_volume.shape)

    # Load camera configurations and foreground masks
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]
    masks = load_foreground_masks()

    # for voxel in voxel_volume:
    for c, config_path in enumerate(camera_configs, start=1):
        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(config_path)
        # voxel_3d = np.array([[voxel[0], voxel[1], voxel[2]]], dtype=np.float32)
        img_points, _ = cv2.projectPoints(voxel_volume, rvec, tvec, camera_matrix, dist_coeffs)
        # x_im, y_im = img_points[0][0]  # Extract image coordinates

        # lookup_table.append((tuple(voxel), c, (x_im, y_im)))
        for voxel, img_point in zip(voxel_volume, img_points):
            x_im, y_im = img_point[0]  # Extract image coordinates
            lookup_table.append((tuple(map(int, voxel)), c, (x_im, y_im)))

    print(len(lookup_table))

    # min_x, max_x = float('inf'), float('-inf')
    # min_y, max_y = float('inf'), float('-inf')

    # # Iterate through the lookup table
    # for _, _, (x_im, y_im) in lookup_table:
    #     # Update the minimum and maximum values
    #     min_x = min(min_x, x_im)
    #     max_x = max(max_x, x_im)
    #     min_y = min(min_y, y_im)
    #     max_y = max(max_y, y_im)

    # # Print out the ranges
    # print(f"x range: {min_x} to {max_x}")
    # print(f"y range: {min_y} to {max_y}")

    # Voxel reconstruction
    data = []
    colors = []
    foreground_threshold = 3  # Number of cameras that need to see the voxel as foreground

     # Voxel reconstruction based on the lookup table
    voxel_visibility = {}  # Tracks the number of cameras that see each voxel as foreground
    for voxel, camera_id, (x_im, y_im) in lookup_table:
        if 0 <= x_im < masks[camera_id].shape[1] and 0 <= y_im < masks[camera_id].shape[0]:  # Check image boundaries
            if masks[camera_id][int(y_im), int(x_im)] > 0:  # Foreground check
                voxel_visibility[voxel] = voxel_visibility.get(voxel, 0) + 1

    # Collect voxels that meet the visibility threshold
    for voxel, count in voxel_visibility.items():
        if count >= foreground_threshold:
            data.append(voxel)    
            colors.append((1.0, 1.0, 1.0))  # Assign a white color for visible voxels

    return data, colors

def main():
    
    world_width = 500
    world_height = 500
    world_depth = 100
    block_size = 3
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]

    # Create a voxel grid with a resolution of 1
#     voxel_volume_bounds = [
#     (-world_width/2 * block_size, world_width/2 * block_size),  # X-axis bounds
#     (-block_size, world_height * block_size),  # Y-axis bounds
#     (-world_depth/2 * block_size, world_depth/2 * block_size)  # Z-axis bounds
# ]

#     resolution = 1

#     voxel_lookup_table = create_lookup_table(voxel_volume_bounds, resolution, camera_configs)

    masks = load_foreground_masks()
    print(masks[1].shape)

    # print(len(voxel_lookup_table))
    # print(masks[1].shape)
    # print([item[-2:] for item in voxel_lookup_table[0:10]])

    # points_of_interest = [(10, 10), (20, 20), (300, 30)]  # Example points

    # for point in points_of_interest:
    #     x, y = point
    #     print(f"Value at ({x}, {y}):", masks[1][y, x])

    # non_zero_pixels = np.nonzero(masks[1])
    # print('Non-zero pixel coordinates:', non_zero_pixels)
    # print('Number of non-zero pixels:', len(non_zero_pixels[0])) 

    

    data, colors = lookup_table = newmethod_lookup(world_width, world_height, world_depth, block_size)
    # data = perform_voxel_reconstruction(lookup_table, masks)

    print("Number of reconstructed voxels:", len(data))
    print("First 10 reconstructed voxels:", data[:10])
    

if __name__ == "__main__":
    main()