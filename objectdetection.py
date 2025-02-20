import numpy as np
import pygame
import math
from rplidar import RPLidar
from sklearn.linear_model import RANSACRegressor

# Set up LIDAR with specified baudrate
PORT_NAME = '/dev/ttyUSB0'  # Change this for your system (e.g., 'COM3' on Windows)
BAUDRATE = 460800  # Your LIDAR's baudrate
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# Set up Pygame
WINDOW_SIZE = (800, 800)
SCALE = 100  # Scale factor for visualization
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("RPLIDAR Room Mapping")

BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (50, 50, 50)
POINT_COLOR = (255, 255, 0)
REFERENCE_COLOR = (0, 255, 0)
LIDAR_COLOR = (0, 0, 255)
WALL_COLOR = (255, 0, 0)

def polar_to_cartesian(distance, angle):
    """Converts polar coordinates (distance, angle) to Cartesian (x, y)."""
    radians = math.radians(angle)
    x = distance * math.cos(radians)
    y = distance * math.sin(radians)
    return x, y

def scan_room():
    """Collects LIDAR data and returns a list of (x, y) coordinates."""
    points = []
    print("Scanning room... Press Ctrl+C to stop.")
    try:
        for scan in lidar.iter_scans():
            for _, angle, distance in scan:
                if 0 < distance < 5000:  # Ignore out-of-range values
                    x, y = polar_to_cartesian(distance / 1000, angle)  # Convert to meters
                    points.append((x, y))
            if len(points) > 1000:  # Limit number of points
                break
    except KeyboardInterrupt:
        print("Stopping scan...")
    return np.array(points)

def detect_reference_corner(points):
    """Finds the bottom-left (min x, min y) corner in the LIDAR scan."""
    if len(points) < 50:
        return None  # Not enough data

    min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
    
    # Get points near the bottom-left
    corner_candidates = points[
        (points[:, 0] < min_x + 0.2) & (points[:, 1] < min_y + 0.2)
    ]
    
    if len(corner_candidates) > 5:
        return np.mean(corner_candidates, axis=0)  # Average for stability
    
    return (min_x, min_y)

def detect_main_wall(points, reference):
    """Finds the dominant vertical wall close to the reference point using RANSAC."""
    near_wall = points[np.abs(points[:, 0] - reference[0]) < 0.5]

    if len(near_wall) < 20:
        return None  # Not enough points for a wall

    # Fit a vertical wall using RANSAC regression
    model = RANSACRegressor()
    model.fit(near_wall[:, 0].reshape(-1, 1), near_wall[:, 1])

    # Predict y values for the estimated wall
    x_wall = np.array([reference[0], reference[0] + 0.1])  # Small segment
    y_wall = model.predict(x_wall.reshape(-1, 1))

    return x_wall, y_wall

def find_lidar_position(points, reference, wall):
    """Finds the LIDAR's position based on distance from the reference point and the wall."""
    if wall is None:
        return None

    wall_x, wall_y = wall
    wall_mid_y = np.mean(wall_y)  # Middle of the detected wall

    # Find point farthest from the wall in X-direction
    max_distance = -1
    lidar_position = None

    for x, y in points:
        distance_to_wall = abs(x - reference[0])  # X-distance to reference
        if distance_to_wall > max_distance:
            max_distance = distance_to_wall
            lidar_position = (x, y)

    return lidar_position

def transform_points(points, reference_point):
    """Shifts all points to make the reference point (0, 0)."""
    return points - np.array(reference_point)

def draw_grid():
    """Draws a grid on the Pygame screen."""
    for i in range(9):
        x = i * SCALE
        y = WINDOW_SIZE[1] - i * SCALE
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, WINDOW_SIZE[1]))
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y))

def draw_points(points, reference, lidar_position, wall):
    """Draws LIDAR points, reference corner, lidar position, and walls in Pygame."""
    screen.fill(BACKGROUND_COLOR)
    draw_grid()

    # Draw LIDAR points
    for x, y in points:
        pygame.draw.circle(screen, POINT_COLOR, (int(x * SCALE), int(WINDOW_SIZE[1] - y * SCALE)), 2)

    # Draw reference corner
    pygame.draw.circle(screen, REFERENCE_COLOR, (int(reference[0] * SCALE), int(WINDOW_SIZE[1] - reference[1] * SCALE)), 6)

    # Draw estimated LIDAR position
    if lidar_position:
        pygame.draw.circle(screen, LIDAR_COLOR, (int(lidar_position[0] * SCALE), int(WINDOW_SIZE[1] - lidar_position[1] * SCALE)), 6)

    # Draw detected wall
    if wall:
        x_wall, y_wall = wall
        pygame.draw.line(screen, WALL_COLOR, 
                         (int(x_wall[0] * SCALE), int(WINDOW_SIZE[1] - y_wall[0] * SCALE)), 
                         (int(x_wall[1] * SCALE), int(WINDOW_SIZE[1] - y_wall[1] * SCALE)), 3)

    pygame.display.flip()

# 🚀 Step 1: Scan the room
lidar_points = scan_room()

# 🚀 Step 2: Detect reference corner
reference_corner = detect_reference_corner(lidar_points)
print(f"Reference Corner Found: {reference_corner}")

# 🚀 Step 3: Detect main wall near the reference
wall = detect_main_wall(lidar_points, reference_corner)
print(f"Main Wall Found: {wall}")

# 🚀 Step 4: Find the actual LIDAR position
lidar_position = find_lidar_position(lidar_points, reference_corner, wall)
print(f"Detected LIDAR Position: {lidar_position}")

# 🚀 Step 5: Transform points to align with reference corner
adjusted_points = transform_points(lidar_points, reference_corner)

# 🚀 Step 6: Display room map in Pygame
draw_points(adjusted_points, (0, 0), lidar_position, wall)

# Pygame event loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            lidar.stop()
            lidar.disconnect()
            pygame.quit()
            exit()
