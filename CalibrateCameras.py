# Used for Task 1 - Calibration
import cv2
import numpy as np
import glob

def calculate_total_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error/len(objpoints)

def draw_3D_axis(ret, mtx, dist, rvecs, tvecs, training_image):
    # Axis points in 3D space. We'll draw the axis lines from the origin to these points.
    # Increasing the numbers makes the lines longer
    axis = np.float32([[0, 0, 0], [9, 0, 0], [0, 9, 0], [0, 0, -9]]).reshape(-1, 3)

    img_with_axes = training_image.copy()

    # Project the 3D points to the 2D image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    imgpts = imgpts.astype(int)

    # Define the origin (chessboard corner in this case)
    origin = tuple(imgpts[0].ravel())
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 5)  # X-Axis in red
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 5)  # Y-Axis in green
    img_with_axes = cv2.line(img_with_axes, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 5)  # Z-Axis in blue

    cv2.imshow('Image_with_axes', img_with_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return imgpts, img_with_axes

def click_event(event, x, y, flags, params):

    global zoom_scale, zoom_window_size
    zoom_scale = 4
    zoom_window_size = 200
    
    if event == cv2.EVENT_LBUTTONDOWN and not original_image is None:
        # Calculate bounds for the zoomed region
        x_min = max(x - zoom_window_size, 0)
        y_min = max(y - zoom_window_size, 0)
        x_max = min(x + zoom_window_size, original_image.shape[1])
        y_max = min(y + zoom_window_size, original_image.shape[0])

        # Extract and zoom in on the region
        zoom_region = original_image[y_min:y_max, x_min:x_max].copy()
        zoom_region = cv2.resize(zoom_region, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)
        
        # Display the zoomed window
        cv2.imshow("Zoomed", zoom_region)
        cv2.setMouseCallback("Zoomed", click_event_zoomed, (x_min, y_min, x_max, y_max))

def click_event_zoomed(event, x, y, flags, params):
    global corner_points, original_image

    if event == cv2.EVENT_LBUTTONDOWN and not original_image is None:
        # Translate click position back to original image coordinates
        x_min, y_min, x_max, y_max = params
        precise_x = int(x / zoom_scale + x_min)
        precise_y = int(y / zoom_scale + y_min)

        # Append precise corner point
        corner_points.append((precise_x, precise_y))
        
        # Visual feedback on the original image
        cv2.circle(original_image, (precise_x, precise_y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", original_image)
        
        if len(corner_points) == 4:
            original_image = None
            cv2.destroyAllWindows()
        else:
            cv2.destroyWindow("Zoomed")

def interpolate_board_points_homography(top_left, top_right, bottom_left, bottom_right, square_size=2.4, rows=6, cols=8):
    
    # Points - image plane
    dst_pts = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")

    # Points - checkerboard plane
    src_pts = np.array([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]], dtype="float32") * square_size

    H, _ = cv2.findHomography(src_pts, dst_pts)

    # Grid points on the checkerboard
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    checkerboard_pts = np.hstack((grid_x.reshape(-1, 1), grid_y.reshape(-1, 1))) * square_size
    checkerboard_pts_homogeneous = np.insert(checkerboard_pts, 2, 1, axis=1).T

    # Show the grid points in the image
    image_pts_homogeneous = np.dot(H, checkerboard_pts_homogeneous)
    image_pts = image_pts_homogeneous[:2, :] / image_pts_homogeneous[2, :]
    
    image_pts = image_pts.T.reshape(-1, 2).astype(np.float32)

    return image_pts

def manual_calibrate(img, square_size):

    # Setting corner_points as global because I cannot pass parameters to the click_event function
    global corner_points, original_image

    # Reset the corner_points for every image
    corner_points = []

    cv2.imshow("img", img)
    original_image = img

    # For images that are partially out of frame
    if(cv2.waitKey(0) == ord('q')):
        return None

    cv2.setMouseCallback("img", click_event, img)
    # cv2.waitKey(0)

    while len(corner_points) < 4:
        print("You did not select the 4 corners. Please try again.")
        cv2.setMouseCallback("img", click_event, img)
        cv2.waitKey(0)

    # We assume that the corner_points are in the following order:
    # top-left, top-right, bottom-left, bottom-right
    top_left = corner_points[0]
    top_right = corner_points[1]
    bottom_left = corner_points[2]
    bottom_right = corner_points[3]

    all_points = interpolate_board_points_homography(top_left, top_right, bottom_left, bottom_right, square_size)
    
    # Draw the points on the image
    for point in all_points:
        cv2.circle(img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    
    # Ensure correct shape (N, 1, 2) - Was needed otherwise in error calculation in main we would get an error
    all_points = all_points.reshape(-1, 1, 2)

    return all_points

def calibrate_camera(images, square_size):

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Some of the below code was taken from here:
    # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for img in images:

        # We make the image grayscale, apply GaussianBlur and then CLAHE to help with the automatic corner detection
        # Overall, it lowers the total error of the calibration
        initial_img = img.copy()

        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 8))

        # Apply CLAHE to the grayscale image
        processed_img = clahe.apply(processed_img)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(processed_img, (8,6), None)
        
        # If found, add object points, image points
        if ret == True:
            print("Found the corners in the image. Proceeding with automatic calibration...")

            # Refines the corner positions - findChessboardCorners uses this function,
            # but we can use it with different parameters to get better results
            corners2 = cv2.cornerSubPix(processed_img, corners, (20,20), (-1,-1), criteria)   

            objpoints.append(objp)
            imgpoints.append(corners2)         

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners2, ret)
            cv2.imshow('img', img)
            # Wait for key press
            # cv2.waitKey(0)

            # If the automatic calibration is not good, we can do manual calibration
            if cv2.waitKey(0) == ord('n'):
                print("Automatic calibration not good. Proceeding with manual calibration...")

                manual_points = manual_calibrate(initial_img, square_size)

                if manual_points is None:
                    continue

                objpoints.pop()
                imgpoints.pop()

                objpoints.append(objp)
                imgpoints.append(manual_points)

                # Draw and display the corners
                cv2.drawChessboardCorners(initial_img, (8,6), manual_points, ret)
                cv2.imshow('img', initial_img)
                # Wait for key press
                cv2.waitKey(0)
        
        # If we didn't find the corners, we will do manual calibration
        else:
            print("Could not find the corners in the image. Proceeding with manual calibration...")

            corners = manual_calibrate(img, square_size)

            if corners is None:
                continue

            objpoints.append(objp)
            imgpoints.append(corners)

            # We set to true to be able to use the corner drawing function
            ret = True
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    return objpoints, imgpoints, processed_img

def extract_frames(video_path, interval=2):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    images = []
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = interval * frame_rate

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Capture frame every 'interval' seconds
        if current_frame % frame_interval == 0:
            images.append(frame)
            # Show the frame for us to see
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
        
        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()
    return images

def main():

    # Probably 11.5
    square_size = 115
    checkerboardWidth = 8
    checkerboardHeight = 6

    video_path = 'data/cam1/intrinsics.avi'
    interval = 2  # Get a frame every 2 seconds
    images = extract_frames(video_path, interval)
    print(images)

    # Read the first image to get its height and width
    img = images[0]

    # Img dimentions 644x486
    height, width = img.shape[:2]
    print("Image dimensions: {}x{}".format(width, height))

    ################### Camera Calibration ###################

    # Call the camera calibration function - automatic calibration
    objpoints, imgpoints, gray_img = calibrate_camera(images, square_size)

    # rvecs and tvecs here are only used to get the error
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

    print("Camera matrix: ")
    print(mtx)
    print("Distortion coefficients: ")
    print(dist)

    # ##################### Online Phase #######################
    
    # # We get the test image, find the corner points, and use the mtx and dist from the camera calibration and the test image points to 
    # # get the rvecs and tvecs. We then use those to create the 3D axis and cube

    # # !!!!!!!! CHANGE ACCORDINGLY TO THE TEST IMAGE YOU WANT TO USE !!!!!!!!
    testing_image_path = 'data/cam1/checkerboard.avi'
    images = extract_frames(testing_image_path, interval=1)

    testing_image = images[0]
    cv2.imshow("Testing image", testing_image)
    cv2.waitKey(0)

    _, imgpoints_test, _ = calibrate_camera(testing_img, square_size)

    print("imgpoints_test: ")
    print(imgpoints_test.shape())

    objp = np.zeros((6*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2) * square_size 

    ret, rvecs_test, tvecs_test = cv2.solvePnP(objp, imgpoints_test, mtx, dist)

    R, _ = cv2.Rodrigues(rvecs_test)

    extrinsic_matrix = np.hstack((R, tvecs_test))

    print("rvecs_test: ")
    print(rvecs_test)
    print("tvecs_test: ")
    print(tvecs_test)

    print("R: ")
    print(R)

    print("Extrinsic matrix: ")
    print(extrinsic_matrix)

    # Checking the error of the calibration
    total_error = calculate_total_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    print("total error: {}".format(total_error))

    # Draw the 3D axis and the cube on the test image
    imgpts, img_with_axes = draw_3D_axis(ret, mtx, dist, rvecs_test, tvecs_test, testing_image)

if __name__ == "__main__":
    main() 