import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

def save_first_frame():

    video_path = "data/cam1/video.avi"
    image_save_path = "data/cam1/first_frame.jpg"

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    
    if ret:
        # If a frame was successfully read, save it to the specified path
        cv2.imwrite(image_save_path, frame)
        print("First frame saved to:", image_save_path)
    else:
        # If no frame was read (e.g., if the video file is empty), print an error message
        print("Failed to capture the first frame from the video.")
    
    # Release the video capture object
    cap.release()

# Used to get our background model
def create_background_model_gmm(video_path):
    cap = cv2.VideoCapture(video_path)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgbg.apply(gray_frame)
    
    # The background model
    background_model = fgbg.getBackgroundImage()
    
    cap.release()
    return background_model

# Automatically find the optimal thresholds for the HSV channels upper bounds
def find_optimal_upper_thresholds(frame_hsv, manual_segmentation):
    best_score = float('inf')
    best_upper_thresholds = None
    
    # Fixed lower bounds at minimum values
    lower_bounds = np.array([0, 0, 0], dtype=np.uint8)

    # Iterate only over the upper bounds for each channel
    for h_high in range(180, 0, -10):
        for s_high in range(255, 0, -10):
            for v_high in range(255, 0, -10):
                
                mask_hue = cv2.inRange(frame_hsv, h_high, 180)
                mask_sat = cv2.inRange(frame_hsv, s_high, 255)
                mask_val = cv2.inRange(frame_hsv, v_high, 255)
                
                combined_mask = cv2.bitwise_and(mask_hue, cv2.bitwise_and(mask_sat, mask_val))
                
                # Compare the mask with the manual segmentation using XOR to evaluate accuracy
                xor_result = cv2.bitwise_xor(combined_mask, manual_segmentation)
                score = np.sum(xor_result)

                # Update the best score and thresholds if the current score is lower
                if score < best_score:
                    best_score = score
                    best_upper_thresholds = (h_high, s_high, v_high)

    return best_upper_thresholds

# Automatically find the optimal thresholds for the HSV channels lower bounds
def find_optimal_lower_thresholds(frame_hsv, manual_segmentation):
    best_score = float('inf')
    best_thresholds = None

    for h_low in range(0, 180, 10):
        for s_low in range(0, 256, 10):
            for v_low in range(0, 256, 10):

                mask_hue = cv2.inRange(frame_hsv, h_low, 180)
                mask_sat = cv2.inRange(frame_hsv, s_low, 255)
                mask_val = cv2.inRange(frame_hsv, v_low, 255)
                
                combined_mask = cv2.bitwise_and(mask_hue, cv2.bitwise_and(mask_sat, mask_val))

                xor_result = cv2.bitwise_xor(combined_mask, manual_segmentation)
                score = np.sum(xor_result)

                if score < best_score:
                    best_score = score
                    best_thresholds = (h_low, s_low, v_low)

    return best_thresholds

# Process the video to extract the foreground mask - We save only the first frame
def process_video(video_path, background_model_gmm, output_subfolder, save_manual_segmentation=False, save_voxel_construction_frame=False):
    os.makedirs(output_subfolder, exist_ok=True)
    
    # Background is already grayscale but with 3 channels so we just make it back to 1 channel
    background_model_gmm = cv2.imread(background_model_gmm)
    background_model_gmm = cv2.cvtColor(background_model_gmm, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(video_path)

    is_first_frame = True
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Make the frame grayscale to match the background model
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        foreground_mask = cv2.absdiff(gray_frame, background_model_gmm)

        _, foreground_mask = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_BINARY)

        # Using foreground mask we get the foreground image on the original frame
        foreground_bgr = cv2.bitwise_and(frame, frame, mask=foreground_mask)

        # change the foreground frame from BGR to HSV
        foreground_hsv = cv2.cvtColor(foreground_bgr, cv2.COLOR_BGR2HSV)

        # Best values found for the lower and upper bounds based on the manual segmentation
        # Camera 1: [0, 0, 10]  [180, 255, 255]
        # Camera 2: [0, 0, 10]  [180, 255, 255]
        # Camera 3: [0, 0, 10]  [180, 255, 255]
        # Camera 4: [0, 0, 10]  [180, 255, 255]

        lower_bounds = [0, 0, 10]
        upper_bounds = [180, 255, 255]

        mask_hue = cv2.inRange(foreground_hsv[:,:,0], lower_bounds[0], upper_bounds[0])
        mask_sat = cv2.inRange(foreground_hsv[:,:,1], lower_bounds[1], upper_bounds[1])
        mask_val = cv2.inRange(foreground_hsv[:,:,2], lower_bounds[2], upper_bounds[2])

        # Combining the masks - example using logical AND to combine thresholds
        combined_mask = cv2.bitwise_and(mask_hue, cv2.bitwise_and(mask_sat, mask_val))

        # kernel1 = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((6, 6), np.uint8)

        # Removing noise.
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel1)

        # Closing small holes.
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel2)

        if is_first_frame:
            if save_manual_segmentation:
                # Save the first frame for manual segmentation
                cv2.imwrite("data/cam4/combined_mask4.jpg", combined_mask)

            if save_voxel_construction_frame:
                # Save the first frame for use in voxel construction
                cv2.imwrite("data/cam3/voxel_construction_frame.jpg", combined_mask) 
                
            is_first_frame = False

        # Display or process the combined_mask as needed
        path = os.path.join(output_subfolder, f"{frame_number}.jpg")
        cv2.imwrite(path, combined_mask)
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Find the optimal thresholds for the HSV channels
    # manual_segmentation_mask = cv2.imread("data/cam1/first_frame4_copy.jpg", cv2.IMREAD_GRAYSCALE)
    # combined_foreground_mask = cv2.imread("data/cam1/combined_mask4.jpg", cv2.IMREAD_GRAYSCALE)
    
    # optimal_lower_thresholds = find_optimal_lower_thresholds(combined_foreground_mask, manual_segmentation_mask)
    # optimal_upper_thresholds = find_optimal_upper_thresholds(combined_foreground_mask, manual_segmentation_mask)
    # print("Optimal thresholds:", optimal_lower_thresholds, "  ", optimal_upper_thresholds)
    
    camera_tasks = [
        ("data/cam1/video.avi", "data/cam1/background_model_gmm.jpg", "output/cam1"),
        ("data/cam2/video.avi", "data/cam2/background_model_gmm.jpg", "output/cam2"),
        ("data/cam3/video.avi", "data/cam3/background_model_gmm.jpg", "output/cam3"),
        ("data/cam4/video.avi", "data/cam4/background_model_gmm.jpg", "output/cam4"),
    ]
    
    # ThreadPoolExecutor to process each camera in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_video, video_path, background_model_path, output_subfolder) for video_path, background_model_path, output_subfolder in camera_tasks]

        # Wait for all tasks to complete
        for future in futures:
            future.result()

    # Used to save the frame for extracting manual segmentation mask
    # save_first_frame()

if __name__ == "__main__":
    main()
