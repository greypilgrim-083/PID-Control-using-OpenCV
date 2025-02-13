import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RobotSimulation:
    def __init__(self):
        # Arena parameters
        self.arena_size = 800
        self.line_width = 30  # 3cm equivalent in pixels
        self.circle_radius = 300  # 30cm equivalent in pixels
        
        # Robot parameters
        self.robot_pos = np.array([400, 400])  # Start at center
        self.robot_angle = 0
        self.robot_speed = 5
        self.sensor_distance = 20
        
        # PID parameters
        self.kp = 0.2
        self.ki = 0.001
        self.kd = 0.1
        self.previous_error = 0
        self.integral = 0
        
        # Create arena
        self.arena = self.create_arena()
        
        # Performance metrics
        self.errors = []
        self.positions = []
        
    def create_arena(self):
        # Create blank arena
        arena = np.ones((self.arena_size, self.arena_size), dtype=np.uint8) * 255
        
        # Draw circular track
        cv2.circle(arena, 
                  (self.arena_size//2, self.arena_size//2), 
                  self.circle_radius, 
                  0, 
                  self.line_width)
        return arena
    
    def get_sensor_reading(self):
        # Calculate sensor position
        sensor_x = int(self.robot_pos[0] + self.sensor_distance * np.cos(self.robot_angle))
        sensor_y = int(self.robot_pos[1] + self.sensor_distance * np.sin(self.robot_angle))
        
        # Ensure within bounds
        sensor_x = np.clip(sensor_x, 0, self.arena_size-1)
        sensor_y = np.clip(sensor_y, 0, self.arena_size-1)
        
        # Get sensor value (0 for black line, 255 for white background)
        return self.arena[sensor_y, sensor_x]
    
    def pid_control(self):
        # Get error (difference from black line)
        error = self.get_sensor_reading() - 0  # Target is 0 (black)
        
        # PID calculations
        p_term = self.kp * error
        self.integral += error
        i_term = self.ki * self.integral
        d_term = self.kd * (error - self.previous_error)
        
        # Update previous error
        self.previous_error = error
        
        # Calculate steering adjustment
        adjustment = -(p_term + i_term + d_term) / 255.0  # Normalize
        
        return adjustment, error
    
    def update(self):
        # Get steering adjustment from PID
        adjustment, error = self.pid_control()
        
        # Update robot angle
        self.robot_angle += adjustment
        
        # Update robot position
        self.robot_pos[0] += self.robot_speed * np.cos(self.robot_angle)
        self.robot_pos[1] += self.robot_speed * np.sin(self.robot_angle)
        
        # Keep robot in bounds
        self.robot_pos = np.clip(self.robot_pos, 0, self.arena_size-1)
        
        # Store metrics
        self.errors.append(error)
        self.positions.append(self.robot_pos.copy())
        
        return self.robot_pos, self.robot_angle

# Visualization
class Visualizer:
    def __init__(self, simulation):
        self.sim = simulation
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        self.scatter = None
        self.line = None
        
    def init(self):
        # Plot arena
        self.ax1.imshow(self.sim.arena, cmap='gray')
        self.scatter = self.ax1.scatter([], [], c='red', s=100)
        self.line, = self.ax2.plot([], [])
        self.ax2.set_ylim(-255, 255)
        self.ax2.set_title('Error over time')
        return self.scatter, self.line
    
    def update(self, frame):
        # Update simulation
        pos, angle = self.sim.update()
        
        # Update robot position plot
        self.scatter.set_offsets(pos)
        
        # Update error plot
        self.line.set_data(range(len(self.sim.errors)), self.sim.errors)
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        return self.scatter, self.line
    
    def animate(self, frames=500):
        anim = FuncAnimation(self.fig, self.update, frames=frames,
                           init_func=self.init, blit=True, interval=20)
        plt.show()

# Run simulation
sim = RobotSimulation()
vis = Visualizer(sim)
vis.animate()

# After simulation, analyze performance
def analyze_performance(simulation):
    positions = np.array(simulation.positions)
    errors = np.array(simulation.errors)
    
    # Calculate metrics
    mean_error = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    
    # Calculate path smoothness (using position derivatives)
    path_smoothness = np.mean(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    
    print(f"Performance Metrics:")
    print(f"Mean Absolute Error: {mean_error:.2f}")
    print(f"Maximum Error: {max_error:.2f}")
    print(f"Path Smoothness (lower is better): {path_smoothness:.2f}")
    
    # Plot final trajectory
    plt.figure(figsize=(10, 10))
    plt.imshow(simulation.arena, cmap='gray')
    plt.plot(positions[:, 0], positions[:, 1], 'r-', label='Robot Path')
    plt.title('Robot Trajectory')
    plt.legend()
    plt.show()

analyze_performance(sim)