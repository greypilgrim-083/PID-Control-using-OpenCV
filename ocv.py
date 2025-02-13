import cv2
import numpy as np
import math
import requests
import urllib.request
import json

class LineDetector:
    def __init__(self, camera_source=None):
        """
        Initialize with different camera sources:
        - None or 0: Default webcam
        - 1,2,etc: Other connected cameras
        - "http://IP:PORT/video": IP camera URL
        - "phone": Will help setup IP Webcam
        """
        self.camera_source = camera_source
        self.setup_camera()
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # ROI parameters
        self.roi_height = 200
        self.roi_top_offset = 100
        
        # Thresholding parameters
        self.threshold_value = 127
        self.max_value = 255
    
    def setup_camera(self):
        """
        Setup camera based on source type
        """
        if self.camera_source == "phone":
            print("\nTo use your phone camera:")
            print("1. Install 'IP Webcam' app from Play Store")
            print("2. Open app and scroll down to 'Start server'")
            print("3. Enter the IP address shown in the app")
            ip = input("Enter IP address (e.g., 192.168.1.100): ")
            self.camera_source = f"http://{ip}:8080/video"
            
        if isinstance(self.camera_source, str) and self.camera_source.startswith("http"):
            # IP camera or phone camera
            self.cap = cv2.VideoCapture(self.camera_source)
        else:
            # Default or USB camera
            self.cap = cv2.VideoCapture(0 if self.camera_source is None else self.camera_source)
        
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera source")
            
    def test_cameras(self):
        """
        Test available camera devices
        """
        available_cameras = []
        for i in range(5):  # Test first 5 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def preprocess_image(self, frame):
        """
        Preprocess the image for line detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, thresh = cv2.threshold(blurred, self.threshold_value, 
                                self.max_value, cv2.THRESH_BINARY_INV)
        
        return thresh
    
    def get_roi(self, frame):
        """
        Extract region of interest
        """
        height = frame.shape[0]
        roi_top = height - self.roi_height - self.roi_top_offset
        roi_bottom = height - self.roi_top_offset
        
        roi = frame[roi_top:roi_bottom, :]
        return roi, roi_top
    
    def detect_line(self, binary_image):
        """
        Detect line in binary image and return center point
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy), largest_contour
        
        return None, None

    def process_frame(self, show_visualization=True):
        """
        Process a single frame and return line position error
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Handle mobile camera rotation if needed
        if isinstance(self.camera_source, str) and self.camera_source.startswith("http"):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Get region of interest
        roi, roi_top = self.get_roi(frame)
        
        # Preprocess image
        binary = self.preprocess_image(roi)
        
        # Detect line
        center_point, contour = self.detect_line(binary)
        
        # Calculate error
        error = 0
        if center_point is not None:
            frame_center = frame.shape[1] // 2
            error = center_point[0] - frame_center
        
        if show_visualization:
            # Draw visualization
            if center_point is not None:
                full_frame_center = (center_point[0], center_point[1] + roi_top)
                cv2.circle(frame, full_frame_center, 5, (0, 255, 0), -1)
                cv2.line(frame, 
                        (frame.shape[1]//2, frame.shape[0]), 
                        full_frame_center, 
                        (255, 0, 0), 2)
                
                if contour is not None:
                    contour_adjusted = contour.copy()
                    contour_adjusted[:,:,1] += roi_top
                    cv2.drawContours(frame, [contour_adjusted], -1, (0, 255, 0), 2)
            
            cv2.rectangle(frame, 
                         (0, roi_top), 
                         (frame.shape[1], roi_top + self.roi_height), 
                         (255, 0, 0), 2)
            
            cv2.putText(frame, f"Error: {error}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Original', frame)
            cv2.imshow('Binary', binary)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None
            elif key == ord('+'):
                self.threshold_value = min(255, self.threshold_value + 5)
            elif key == ord('-'):
                self.threshold_value = max(0, self.threshold_value - 5)
        
        return error
    
    def run(self):
        """
        Main loop for continuous processing
        """
        try:
            while True:
                error = self.process_frame()
                if error is None:
                    break
                
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.cap.release()
        cv2.destroyAllWindows()

def find_available_cameras():
    """
    Find and test all available cameras
    """
    print("Testing available cameras...")
    
    # Test USB/built-in cameras
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(f"Camera {i}")
            cap.release()
    
    return available_cameras

def main():
    # Show available options
    print("\nCamera Options:")
    print("1. Use default webcam")
    print("2. Use phone camera")
    print("3. List available cameras")
    print("4. Enter custom IP camera URL")
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        detector = LineDetector()
    elif choice == '2':
        detector = LineDetector("phone")
    elif choice == '3':
        cameras = find_available_cameras()
        if not cameras:
            print("No cameras found!")
            return
        print("\nAvailable cameras:")
        for i, camera in enumerate(cameras):
            print(f"{i+1}. {camera}")
        cam_choice = int(input("\nSelect camera number: ")) - 1
        detector = LineDetector(cam_choice)
    elif choice == '4':
        url = input("Enter camera URL: ")
        detector = LineDetector(url)
    else:
        print("Invalid choice!")
        return
    
    print("\nControls:")
    print("  '+' : Increase threshold")
    print("  '-' : Decrease threshold")
    print("  'q' : Quit")
    
    detector.run()

if __name__ == "__main__":
    main()