import pygame
import math
import numpy as np
from rplidar import RPLidar

# ----- Configuration -----
PORT_NAME = '/dev/ttyUSB0'     # Change to your LIDAR port (e.g., 'COM3' on Windows)
BAUDRATE = 460800              # RPLidar C1 baudrate
ROOM_SIZE = 8.0                # Room dimensions: 8x8 meters
SCALE = 100                    # Scale for visualization: 100 pixels per meter

# ----- Initialize RPLidar -----
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# ----- Initialize Pygame -----
WINDOW_SIZE = (int(ROOM_SIZE * SCALE), int(ROOM_SIZE * SCALE))
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Mapping & Localization Using Room Walls")

# Colors for visualization
BACKGROUND_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)       # Walls drawn in red
ROBOT_COLOR = (0, 255, 0)      # Robot position drawn in green

def get_wall_distance(scan, target_angle, angle_tolerance=10):
    """
    Computes the average distance (in meters) for LIDAR measurements
    within target_angle ± angle_tolerance.
    
    For example, use target_angle=180 for the back wall and 270 for the right wall.
    """
    distances = []
    for quality, angle, distance in scan:
        # Check if angle is within the target sector (accounting for wrap-around if needed)
        if abs(angle - target_angle) < angle_tolerance:
            # Only consider valid nonzero measurements (distance in mm)
            if distance > 0:
                distances.append(distance / 1000.0)  # Convert mm to meters
    if distances:
        return np.mean(distances)
    return None

def draw_map(robot_pos):
    """
    Draw the room boundaries and the robot's estimated position.
    
    The room is assumed to have walls at x=0, x=ROOM_SIZE and y=0, y=ROOM_SIZE.
    Coordinates are converted to screen space (with (0,0) at the bottom left).
    """
    screen.fill(BACKGROUND_COLOR)
    
    # Draw the room boundary rectangle
    # (Global coordinates: bottom-left (0,0) to top-right (ROOM_SIZE, ROOM_SIZE))
    rect = pygame.Rect(0, 0, ROOM_SIZE * SCALE, ROOM_SIZE * SCALE)
    pygame.draw.rect(screen, WALL_COLOR, rect, 2)
    
    # Draw the robot as a circle
    rx, ry = robot_pos
    screen_x = int(rx * SCALE)
    # In Pygame, y=0 is at the top so we invert the y coordinate
    screen_y = int(WINDOW_SIZE[1] - ry * SCALE)
    pygame.draw.circle(screen, ROBOT_COLOR, (screen_x, screen_y), 5)
    
    pygame.display.flip()

def main():
    print("Mapping & Localization started. Close the window to stop.")
    try:
        while True:
            # Get a single LIDAR scan
            for scan in lidar.iter_scans():
                # Extract distance to the back wall (assumed at x=0, measured near 180°)
                d_back = get_wall_distance(scan, 180, angle_tolerance=10)
                # Extract distance to the right wall (assumed at y=0, measured near 270°)
                d_right = get_wall_distance(scan, 270, angle_tolerance=10)
                
                if d_back is not None and d_right is not None:
                    # Assuming the sensor is oriented with the room,
                    # its global position is:
                    # x coordinate = measured distance to the back wall
                    # y coordinate = measured distance to the right wall
                    robot_pos = (d_back, d_right)
                    print("Estimated Robot Position (meters):", robot_pos)
                    
                    draw_map(robot_pos)
                
                # Check for Pygame quit events to exit cleanly
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                # Break out after one scan per loop iteration
                break
                
    except KeyboardInterrupt:
        print("Stopping mapping and localization.")
    finally:
        lidar.stop()
        lidar.disconnect()
        pygame.quit()

if __name__ == "__main__":
    main()
