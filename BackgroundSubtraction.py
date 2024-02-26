import cv2
import numpy as np

import cv2

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

def process_video(video_path, background_model_gmm):

    # Background is already grayscale but with 3 channels so we just make it back to 1 channel
    background_model_gmm = cv2.cvtColor(background_model_gmm, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(video_path)

    is_first_frame = True

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

        #kernel1 = np.ones((4, 4), np.uint8)
        kernel2 = np.ones((6, 6), np.uint8)

        # Erode and then dilate, known as opening. Good for removing noise.
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel1)

        # Dilate and then erode, known as closing. Good for closing small holes.
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel2)

        # if is_first_frame:
        #     # Save the first frame for manual segmentation
        #     cv2.imwrite("data/cam4/combined_mask4.jpg", combined_mask)
        #     is_first_frame = False

        if is_first_frame:
            # Save the frame for use in voxel construction
            cv2.imwrite("data/cam3/voxel_construction_frame.jpg", combined_mask) 
            is_first_frame = False

        # Display or process the combined_mask as needed
        cv2.imshow('Foreground Mask', combined_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():

    # background_video_path = "data/cam1/background.avi"

    # background_model_gmm = create_background_model_gmm(background_video_path)

    # cv2.imwrite("data/cam4/background_model_gmm.jpg", background_model_gmm)

    manual_segmentation_mask = cv2.imread("data/cam4/first_frame4_copy.jpg", cv2.IMREAD_GRAYSCALE)
    combined_foreground_mask = cv2.imread("data/cam4/combined_mask4.jpg", cv2.IMREAD_GRAYSCALE)

    # Find the optimal thresholds for the HSV channels
    # optimal_lower_thresholds = find_optimal_lower_thresholds(combined_foreground_mask, manual_segmentation_mask)
    # optimal_upper_thresholds = find_optimal_upper_thresholds(combined_foreground_mask, manual_segmentation_mask)
    # print("Optimal thresholds:", optimal_lower_thresholds, "  ", optimal_upper_thresholds)

    process_video_path1 = "data/cam1/video.avi"
    background_model_gmm1 = cv2.imread("data/cam1/background_model_gmm.jpg")

    process_video_path2 = "data/cam2/video.avi"
    background_model_gmm2 = cv2.imread("data/cam2/background_model_gmm.jpg")

    process_video_path3 = "data/cam3/video.avi"
    background_model_gmm3 = cv2.imread("data/cam3/background_model_gmm.jpg")

    process_video_path4 = "data/cam4/video.avi"
    background_model_gmm4 = cv2.imread("data/cam4/background_model_gmm.jpg")

    process_video(process_video_path3, background_model_gmm3)

    # Used to save the frame for extracting manual segmentation mask
    # save_first_frame()

if __name__ == "__main__":
    main()