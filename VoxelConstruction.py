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
            lookup_table.append((*voxel, c, xim, yim))  # Append voxel coord, camera index, and img coord

    return lookup_table

def main():
    
    camera_configs = ["data/cam1/config.xml", "data/cam2/config.xml", "data/cam3/config.xml", "data/cam4/config.xml"]
    # camera_parameters = []
    
    # Load camera parameters
    # for config_path in camera_configs:
    #     camera_matrix, dist_coeffs, rvec, tvec = load_camera_parameters(config_path)
    #     camera_parameters.append({
    #         "camera_matrix": camera_matrix,
    #         "dist_coeffs": dist_coeffs,
    #         "rvec": rvec,
    #         "tvec": tvec
    #     })

    voxel_volume_bounds = [(-1, 1), (-1, 1), (0, 2)]  # Define your voxel volume bounds
    resolution = 0.01  # Define the resolution of your voxel grid

    voxel_lookup_table = create_lookup_table(voxel_volume_bounds, resolution, camera_configs)

    print(voxel_lookup_table[0:10])

if __name__ == "__main__":
    main()