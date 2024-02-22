import cv2
import numpy as np

class BackgroundSubtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.initialize_model()

    def initialize_model(self):
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Update MOG2 model with each frame
            self.mog2.apply(frame)
        cap.release()

    def subtract_background(self, frame):
        # Use MOG2 model to get the foreground mask
        foreground_mask = self.mog2.apply(frame, learningRate=-1)  # No model update
        return foreground_mask

    @staticmethod
    def post_process(image):
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        return dilated

    @staticmethod
    def overlay_mask_on_frame(frame, mask):
        # Convert mask to a 3 channel image so it can be colorized
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # Colorize the mask in green
        colored_mask[mask == 255] = [0, 255, 0]
        # Overlay the mask on the original frame
        overlay_frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
        return overlay_frame

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Subtract background
            foreground_mask = self.subtract_background(frame)
            # Post-processing
            cleaned_mask = self.post_process(foreground_mask)
            # Overlay mask on the original frame for visualization
            overlay_frame = self.overlay_mask_on_frame(frame, cleaned_mask)
            # Show the overlay frame
            cv2.imshow('Foreground Overlay', overlay_frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    background_video_path = 'data/cam1/background.avi'
    video_path = 'data/cam1/video.avi'
    
    subtractor = BackgroundSubtractor(background_video_path)
    subtractor.process_video(video_path)
