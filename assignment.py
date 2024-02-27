import glm
import random
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import VoxelConstruction as vc

block_size = 1

camera_configs = [
        "data/cam1/config.xml",
        "data/cam2/config.xml",
        "data/cam3/config.xml",
        "data/cam4/config.xml"
    ]

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

    x_range = np.linspace(-600, 900, num=100)
    y_range = np.linspace(-800, 800, num=100)
    z_range = np.linspace(-2000, 600, num=100)
    voxel_volume = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

    lookup_table = []

    # Load camera configurations and foreground masks
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]
    masks = vc.load_foreground_masks()

    for c, config_path in enumerate(camera_configs, start=1):
        camera_matrix, dist_coeffs, rvec, tvec = vc.load_camera_parameters(config_path)
        # voxel_3d = np.array([[voxel[0], voxel[1], voxel[2]]], dtype=np.float32)
        img_points, _ = cv2.projectPoints(voxel_volume, rvec, tvec, camera_matrix, dist_coeffs)
        # x_im, y_im = img_points[0][0]  # Extract image coordinates

        # lookup_table.append((tuple(voxel), c, (x_im, y_im)))
        for voxel, img_point in zip(voxel_volume, img_points):
            x_im, y_im = img_point[0]  # Extract image coordinates
            lookup_table.append((tuple(map(int, voxel)), c, (x_im, y_im)))

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

    scale_x = 128 / (900 + 600)  # Adjust based on actual desired range and real range
    scale_y = 64 / (800 + 800)   # Same here
    scale_z = 128 / (2000 + 600) # And here

    # Collect voxels that meet the visibility threshold
    for voxel, count in voxel_visibility.items():
        if count >= foreground_threshold:
            scaled_x = (voxel[0] + 600) * scale_x - width / 2  # Recenter after scaling
            scaled_y = (voxel[1] + 800) * scale_y - height / 2  # Adjust centering as needed
            scaled_z = (voxel[2] + 2000) * scale_z - depth / 2  # Adjust centering as needed
            data.append([scaled_x, scaled_y, scaled_z])
            colors.append((1.0, 1.0, 1.0))  # Assign a white color for visible voxels

    return data, colors

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    
    cam_positions = []
    for config_path in camera_configs:
        rotation_matrix = load_rotation_matrix_from_xml(config_path)
        translation_vector = load_translation_from_xml(config_path)
        position_vector = -np.matrix(rotation_matrix).T * np.matrix(translation_vector).T

        cam_positions.append([position_vector[0], -position_vector[2], position_vector[1]])

    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

   
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    # return cam_rotations

    cam_rotations = []
    for config_path in camera_configs:
        R_mat = load_rotation_matrix_from_xml(config_path)
        
        # Convert the numpy rotation matrix to a glm matrix
    #     glm_rotation_matrix = glm.mat4(
    #         rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2], 0,
    #         rotation_matrix[1,0], rotation_matrix[1,1], rotation_matrix[1,2], 0,
    #         rotation_matrix[2,0], rotation_matrix[2,1], rotation_matrix[2,2], 0,
    #         0, 0, 0, 1
    #     )
        
    #     cam_rotations.append(glm_rotation_matrix)
    
    # return cam_rotations
        
     # Convert to OpenGL coordinate system
        R_mat_gl = R_mat[:, [0, 2, 1]]
        R_mat_gl[1, :] *= -1  # Invert the y-axis for OpenGL's coordinate system
        
        # Create a 4x4 OpenGL-compatible rotation matrix
        gl_rot_mat = np.eye(4)
        gl_rot_mat[:3, :3] = R_mat_gl

        # Convert to glm matrix for further transformations if necessary
        gl_rot_mat_glm = glm.mat4(*gl_rot_mat.T.ravel())
        
        # Adjust for any additional rotation as necessary
        rotation_matrix_y = glm.rotate(glm.mat4(1), glm.radians(-90), glm.vec3(0, 1, 0))
        cam_rotation = gl_rot_mat_glm * rotation_matrix_y
        
        cam_rotations.append(cam_rotation)
        
    return cam_rotations

