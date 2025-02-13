import cv2
import numpy as np
import time
import sys

try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY_PI = True
except (ImportError, RuntimeError):
    IS_RASPBERRY_PI = False

class PIDController:
    """Improved PID controller with anti-windup and clamping"""
    def __init__(self, kp=0.25, ki=0.01, kd=0.15, max_integral=50, output_limits=(-50, 50)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.output_limits = output_limits
        self.reset()
        
    def reset(self):
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Proportional term
        p = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        i = self.ki * self.integral
        
        # Derivative term
        d = self.kd * (error - self.prev_error) / dt if dt > 0 else 0
        self.prev_error = error
        
        # Calculate and clamp output
        output = p + i + d
        return np.clip(output, *self.output_limits)

class MotorDriver:
    """Hardware-abstracted motor controller"""
    def __init__(self, config):
        self.config = config
        self.simulation_mode = not IS_RASPBERRY_PI
        
        if not self.simulation_mode:
            GPIO.setmode(GPIO.BCM)
            self._setup_pins()
            
    def _setup_pins(self):
        """Initialize GPIO pins and PWM"""
        pins = [
            self.config['left_forward'],
            self.config['left_backward'],
            self.config['right_forward'],
            self.config['right_backward'],
            self.config['left_pwm'],
            self.config['right_pwm']
        ]
        
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
        
        self.left_pwm = GPIO.PWM(self.config['left_pwm'], 1000)
        self.right_pwm = GPIO.PWM(self.config['right_pwm'], 1000)
        self.left_pwm.start(0)
        self.right_pwm.start(0)
        
    def set_speed(self, left, right):
        """Set motor speeds with range -100 to 100"""
        left = np.clip(left, -100, 100)
        right = np.clip(right, -100, 100)
        
        if self.simulation_mode:
            print(f"Motors: L={left:.1f}% R={right:.1f}%")
            return

        # Left motor control
        GPIO.output(self.config['left_forward'], GPIO.HIGH if left >=0 else GPIO.LOW)
        GPIO.output(self.config['left_backward'], GPIO.LOW if left >=0 else GPIO.HIGH)
        self.left_pwm.ChangeDutyCycle(abs(left))

        # Right motor control
        GPIO.output(self.config['right_forward'], GPIO.HIGH if right >=0 else GPIO.LOW)
        GPIO.output(self.config['right_backward'], GPIO.LOW if right >=0 else GPIO.HIGH)
        self.right_pwm.ChangeDutyCycle(abs(right))
        
    def cleanup(self):
        """Stop motors and cleanup resources"""
        self.set_speed(0, 0)
        if not self.simulation_mode:
            self.left_pwm.stop()
            self.right_pwm.stop()
            GPIO.cleanup()

class LineDetector:
    """Line detection using OpenCV with custom camera support"""
    def __init__(self, camera_source=0):
        self.camera_source = camera_source
        self.cap = self._initialize_camera()
        
        if not self.cap.isOpened():
            raise ValueError("Failed to open camera source")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # ROI parameters
        self.roi_height = 200
        self.roi_top_offset = 100
        
        # Thresholding parameters
        self.threshold_value = 127
        self.max_value = 255
    
    def _initialize_camera(self):
        """Initialize camera based on source type"""
        if isinstance(self.camera_source, str):
            if self.camera_source.lower() == "phone":
                print("\nTo use your phone camera:")
                print("1. Install 'IP Webcam' app from Play Store")
                print("2. Open app and scroll down to 'Start server'")
                print("3. Enter the IP address shown in the app")
                ip = input("Enter IP address (e.g., 192.168.1.100): ")
                self.camera_source = f"http://{ip}:8080/video"
            
            # Open IP camera or phone camera
            return cv2.VideoCapture(self.camera_source)
        else:
            # Default or USB camera
            return cv2.VideoCapture(int(self.camera_source))
    
    def preprocess_image(self, frame):
        """Preprocess the image for line detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Binary threshold
        _, thresh = cv2.threshold(blurred, self.threshold_value, 
                                self.max_value, cv2.THRESH_BINARY_INV)
        
        return thresh
    
    def get_roi(self, frame):
        """Extract region of interest"""
        height = frame.shape[0]
        roi_top = height - self.roi_height - self.roi_top_offset
        roi_bottom = height - self.roi_top_offset
        
        roi = frame[roi_top:roi_bottom, :]
        return roi, roi_top
    
    def detect_line(self, binary_image):
        """Detect line in binary image and return center point"""
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
        """Process a single frame and return line position error with enhanced debugging"""
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Get region of interest
        roi, roi_top = self.get_roi(frame)
        
        # Preprocess image with intermediate visualization
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.threshold_value, 
                                self.max_value, cv2.THRESH_BINARY_INV)
        
        # Detect line
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate error
        error = 0
        center_point = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                center_point = (cx, cy)
                frame_center = frame.shape[1] // 2
                error = cx - frame_center
        
        if show_visualization:
            # Create debug composite
            debug_row1 = np.hstack([gray, blurred])
            debug_row1 = cv2.cvtColor(debug_row1, cv2.COLOR_GRAY2BGR)
            
            # Create color binary image for visualization
            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # Draw contours on binary image if found
            if contours:
                cv2.drawContours(binary_color, contours, -1, (0, 255, 0), 2)
                
            # Add text labels
            cv2.putText(debug_row1, 'Grayscale', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_row1, 'Blurred', (gray.shape[1] + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw visualization on main frame
            if center_point is not None:
                full_frame_center = (center_point[0], center_point[1] + roi_top)
                cv2.circle(frame, full_frame_center, 5, (0, 255, 0), -1)
                cv2.line(frame, 
                        (frame.shape[1]//2, frame.shape[0]), 
                        full_frame_center, 
                        (255, 0, 0), 2)
            
            cv2.rectangle(frame, 
                        (0, roi_top), 
                        (frame.shape[1], roi_top + self.roi_height), 
                        (255, 0, 0), 2)
            
            # Add threshold value and error display
            cv2.putText(frame, f"Threshold: {self.threshold_value}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Error: {error}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show all windows
            cv2.imshow('Processing Steps', debug_row1)
            cv2.imshow('Binary with Contours', binary_color)
            cv2.imshow('Original', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None
            elif key == ord('+'):
                self.threshold_value = min(255, self.threshold_value + 5)
            elif key == ord('-'):
                self.threshold_value = max(0, self.threshold_value - 5)
        
        return error
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
#192.168.209.219:8080
class LineFollowerRobot:
    """Main robot control class integrating detection and motion"""
    def __init__(self, motor_config, pid_params, camera_source=0):
        self.detector = LineDetector(camera_source)
        self.motors = MotorDriver(motor_config)
        self.pid = PIDController(**pid_params)
        
        # Control parameters
        self.base_speed = 40
        self.running = True
        
    def run(self):
        """Main control loop"""
        try:
            while self.running:
                # Get line position error
                error = self.detector.process_frame(show_visualization=True)
                
                if error is None:  # Quit signal
                    break
                    
                # PID control
                control = self.pid.compute(error)
                
                # Motor mixing
                left_speed = self.base_speed - control
                right_speed = self.base_speed + control
                
                self.motors.set_speed(left_speed, right_speed)
                
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Release all resources"""
        self.motors.cleanup()
        self.detector.cleanup()

# Configuration
motor_config = {
    "left_forward": 17,
    "left_backward": 18,
    "right_forward": 22,
    "right_backward": 23,
    "left_pwm": 19,
    "right_pwm": 26
}

pid_params = {
    "kp": 0.25,
    "ki": 0.01,
    "kd": 0.15,
    "max_integral": 50,
    "output_limits": (-50, 50)
}

if __name__ == "__main__":
    try:
        print("Starting line following robot...")
        print("Camera Options:")
        print("1. Use default webcam")
        print("2. Use phone camera")
        print("3. Enter custom IP camera URL")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            camera_source = 0
        elif choice == '2':
            camera_source = "phone"
        elif choice == '3':
            camera_source = input("Enter IP camera URL: ")
        else:
            print("Invalid choice! Using default webcam.")
            camera_source = 0
        
        robot = LineFollowerRobot(motor_config, pid_params, camera_source)
        robot.run()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)