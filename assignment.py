import glm
import random
import numpy as np
import xml.etree.ElementTree as ET
import cv2

block_size = 1

camera_configs = [
        "data/cam1/config.xml",
        "data/cam2/config.xml",
        "data/cam3/config.xml",
        "data/cam4/config.xml"
    ]

def load_camera_parameters(camera_config_path):
    fs = cv2.FileStorage(camera_config_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    dist_coeffs = fs.getNode("DistortionCoeffs").mat()
    rvec = fs.getNode("RotationVectors").mat()
    tvec = fs.getNode("TranslationVectors").mat()
    fs.release()
    return camera_matrix, dist_coeffs, rvec, tvec

# Load original camera images for coloring the voxels
def load_camera_images():
    images = {}
    for cam_id in range(1, 5):
        image_path = f"data/cam{cam_id}/first_frame.jpg"
        images[cam_id] = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return images

def load_foreground_masks():
    masks = {}
    for cam_id in range(1, 5):
        mask_path = f"data/cam{cam_id}/voxel_construction_frame.jpg"
        masks[cam_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return masks

def load_rotation_matrix_from_xml(config_path):
    # Parse the XML to get the rotation matrix
    tree = ET.parse(config_path)
    root = tree.getroot()
    rotation_matrix_data = root.find('RotationMatrix/data').text.strip().split()
    rotation_matrix_data = [float(num) for num in rotation_matrix_data]
    # Assuming the rotation matrix is stored row-wise
    rotation_matrix = np.array(rotation_matrix_data).reshape((3, 3))
    return rotation_matrix

def load_translation_from_xml(config_path):
    # Parse the XML to get the translation vector
    tree = ET.parse(config_path)
    root = tree.getroot()
    translation_data = root.find('TranslationVectors/data').text.strip().split()
    translation_data = [float(num) for num in translation_data]
    return translation_data

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):

    # Used hardcoded values instead of the input parameters
    # These numbers seem to capture almost all of the 2D image dimentions (486x644)
    x_range = np.linspace(-512, 1024, num=100)
    y_range = np.linspace(-1024, 1024, num=100)
    z_range = np.linspace(-2048, 512, num=100)
    voxel_volume = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    ##### Look up table #####
    lookup_table = []

    # Load camera configurations and foreground masks
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]
    masks = load_foreground_masks()
    colored_rgb_images = load_camera_images()

    # We can give the voxel volume and get back a list of the image points
    for c, config_path in enumerate(camera_configs, start=1):

        camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(config_path)
        img_points, _ = cv2.projectPoints(voxel_volume, rvec, tvec, camera_matrix, dist_coeffs)

        # Align the voxel and camera points and append to the lookup table
        # Make voxels to integers
        for voxel, img_point in zip(voxel_volume, img_points):
            x_im, y_im = img_point[0]  # Extract image coordinates
            lookup_table.append((tuple(map(int, voxel)), c, (x_im, y_im)))

    ##### Voxel reconstruction #####
    data = []
    colors = []
    # Number of cameras that need to see the voxel as foreground
    foreground_threshold = 4

    # Tracks the number of cameras that see each voxel as foreground
    voxel_visibility = {}

    # Store the cumulative colors and count for averaging
    voxel_colors = {}

    for voxel, camera_id, (x_im, y_im) in lookup_table:
        # Check image boundaries
        if 0 <= x_im < masks[camera_id].shape[1] and 0 <= y_im < masks[camera_id].shape[0]:
            # If that 2D pixel is foreground then increment the voxel visibility
            if masks[camera_id][int(y_im), int(x_im)] > 0:  
                voxel_visibility[voxel] = voxel_visibility.get(voxel, 0) + 1

                if camera_id == 1:
                    # Extract color from the camera image
                    color = colored_rgb_images[camera_id][int(y_im), int(x_im), :]
                    if voxel not in voxel_colors:
                        voxel_colors[voxel] = (np.array(color), 1)  # Store color and a count
                    else:
                        # Redundant currently but could be used to average colors in future
                        # Accumulate colors and increment count
                        voxel_colors[voxel] = (voxel_colors[voxel][0] + np.array(color), voxel_colors[voxel][1] + 1)

    # Scale the voxel positions to fit the 128x64x128 grid
    simple_scale = 64

    # Collect voxels that meet the visibility threshold
    for voxel, count in voxel_visibility.items():
        if count >= foreground_threshold:

            scaled_x = voxel[0] / simple_scale
            scaled_y = - (voxel[2] / simple_scale )
            scaled_z = voxel[1] / simple_scale
            data.append([scaled_x, scaled_y, scaled_z])
            
            # Calculate average color
            if voxel in voxel_colors:
                avg_color = voxel_colors[voxel][0] / voxel_colors[voxel][1]
                # Without this, color is white
                avg_color = avg_color[::-1] / 255.0  
                colors.append(avg_color)
            else:
                colors.append((1.0, 1.0, 1.0))

    return data, colors

def get_cam_positions():
    def process_camera(camera_id):
        config_path = f"data/cam{camera_id}/config.xml"
        _, _, rvec, tvec = load_camera_parameters(config_path)
        rmtx, _ = cv2.Rodrigues(rvec)
        position = np.dot(-rmtx.T, np.array(tvec).reshape(-1, 1))
        return np.array([position[0, 0], abs(position[2, 0]), position[1, 0]]) / 10

    return [process_camera(camera_id) for camera_id in range(1, 5)], np.eye(3).tolist() + [[1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    cam_rotations = []
    for cam_index in range(1, 5):
        config_path = f"data/cam{cam_index}/config.xml"
        _, _, rotation_vector, _ = load_camera_parameters(config_path)
        unit_matrix = np.eye(4)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        unit_matrix[:3, :3] = rotation_matrix
        matrix = glm.mat4(unit_matrix)
        matrix = glm.rotate(matrix, glm.radians(90), glm.vec3(0, 1, 1))
        cam_rotations.append(matrix)
    return cam_rotations
