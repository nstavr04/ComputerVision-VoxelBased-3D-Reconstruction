import glm
import random
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import VoxelConstruction as vc

block_size = 1.0

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

    voxel_volume_bounds = [
    (-width/2 * block_size, width/2 * block_size),  # X-axis bounds
    (-block_size, height * block_size),  # Y-axis bounds
    (-depth/2 * block_size, depth/2 * block_size)  # Z-axis bounds
    ]
    resolution = 1

    voxel_lookup_table = vc.create_lookup_table(voxel_volume_bounds, resolution, camera_configs)
    masks = vc.load_foreground_masks()

    reconstructed_voxels = vc.perform_voxel_reconstruction(voxel_lookup_table, masks)

    data, colors = [], []
    for voxel in reconstructed_voxels:
        x, y, z = voxel
        data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
        colors.append([x / width, z / depth, y / height])
    return data, colors

def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    
    cam_positions = []
    for config_path in camera_configs:
        rotation_matrix = load_rotation_matrix_from_xml(config_path)
        translation_vector = load_translation_from_xml(config_path)
        position_vector = -np.matrix(rotation_matrix).T * np.matrix(translation_vector).T

        cam_positions.append([position_vector[0][0], -position_vector[2][0], position_vector[1][0]])

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
        rotation_matrix = load_rotation_matrix_from_xml(config_path)
        
        # Convert the numpy rotation matrix to a glm matrix
        glm_rotation_matrix = glm.mat4(
            rotation_matrix[0,0], rotation_matrix[0,1], rotation_matrix[0,2], 0,
            rotation_matrix[1,0], rotation_matrix[1,1], rotation_matrix[1,2], 0,
            rotation_matrix[2,0], rotation_matrix[2,1], rotation_matrix[2,2], 0,
            0, 0, 0, 1
        )
        
        cam_rotations.append(glm_rotation_matrix)
    
    return cam_rotations

