# confirmation
import threading
import pygame
import math
import numpy as np
from rplidar import RPLidar
from sklearn.cluster import DBSCAN
import time
import heapq

# ----- Configuration -----
PORT_NAME = '/dev/ttyUSB0'      # Change to your LIDAR port (e.g., 'COM3' on Windows)
BAUDRATE = 460800               # RPLidar C1 baudrate
ROOM_SIZE = 8.0                 # Room dimensions: 8x8 meters
SCALE = 100                     # Scale for visualization: 100 pixels per meter
WALL_MARGIN = 0.2               # Margin (meters) for filtering wall-adjacent points
UPSIDE_DOWN = True              # Set True if LIDAR is mounted upside down
CELL_SIZE = 0.1                 # Grid resolution for pathfinding (meters)
DETECTION_DURATION = 5.0        # Scanning duration (seconds) for the detection phase

# ----- Determine Target Angles Based on Orientation -----
# For a normally mounted LIDAR: use 180° for the back wall and 270° for the right wall.
# For an upside down LIDAR: the right wall is seen at 90°.
BACK_TARGET = 180
RIGHT_TARGET = 90 if UPSIDE_DOWN else 270

# ----- Global Variables -----
latest_robot_pos = None      # Continuously updated robot position (meters)
accumulated_points = []      # Accumulated scan points for object detection
scanning_active = True       # Flag to control accumulation during detection phase

# ----- Initialize LIDAR -----
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# ----- Initialize Pygame -----
WINDOW_SIZE = (int(ROOM_SIZE * SCALE), int(ROOM_SIZE * SCALE))
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Mapping, Localization & Path Planning (Upside Down)")

# Colors for visualization
BACKGROUND_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)       # Red for walls
ROBOT_COLOR = (0, 255, 0)      # Green for the robot
OBJECT_COLOR = (0, 0, 255)     # Blue for detected objects
PATH_COLOR = (255, 255, 0)     # Yellow for planned paths

# ----- LIDAR Processing Functions -----
def get_wall_distance(scan, target_angle, angle_tolerance=10):
    """
    Computes the average distance (meters) for measurements within target_angle ± angle_tolerance.
    """
    distances = [d / 1000.0 for _, angle, d in scan 
                 if abs(angle - target_angle) < angle_tolerance and d > 0]
    return np.mean(distances) if distances else None

def get_global_points(robot_pos, scan):
    """
    Converts LIDAR polar coordinates into global Cartesian coordinates.
    Adjusts for an upside-down mounted LIDAR.
    """
    points = []
    for _, angle, distance in scan:
        if distance > 0:
            d = distance / 1000.0
            rad = math.radians(angle)
            x_local = d * math.cos(rad)
            y_local = -d * math.sin(rad) if UPSIDE_DOWN else d * math.sin(rad)
            global_x = robot_pos[0] + x_local
            global_y = robot_pos[1] + y_local
            points.append((global_x, global_y))
    return np.array(points)

def filter_interior_points(points, room_size, margin):
    """Removes points that are too close to the walls."""
    return np.array([p for p in points if margin < p[0] < (room_size - margin) and margin < p[1] < (room_size - margin)])

def cluster_objects(points, eps=0.2, min_samples=3):
    """Clusters points using DBSCAN and returns cluster centroids."""
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return [np.mean(points[labels == label], axis=0) for label in set(labels) if label != -1]

# ----- Drawing Function -----
def draw_map(robot_pos, objects, path=None):
    """
    Draws room boundaries, robot position, detected objects, and planned path.
    """
    screen.fill(BACKGROUND_COLOR)
    
    # Draw room boundary
    pygame.draw.rect(screen, WALL_COLOR, pygame.Rect(0, 0, ROOM_SIZE * SCALE, ROOM_SIZE * SCALE), 2)
    
    # Draw robot position (if available)
    if robot_pos:
        rx, ry = robot_pos
        screen_x = int(rx * SCALE)
        screen_y = int(WINDOW_SIZE[1] - ry * SCALE)  # Invert y-axis for Pygame
        pygame.draw.circle(screen, ROBOT_COLOR, (screen_x, screen_y), 5)
    
    # Draw detected objects
    for obj in objects:
        ox, oy = obj
        obj_screen = (int(ox * SCALE), int(WINDOW_SIZE[1] - oy * SCALE))
        pygame.draw.circle(screen, OBJECT_COLOR, obj_screen, 4)
    
    # Draw planned path (if available)
    if path and len(path) > 1:
        path_points = [(int(x * SCALE), int(WINDOW_SIZE[1] - y * SCALE)) for x, y in path]
        pygame.draw.lines(screen, PATH_COLOR, False, path_points, 2)
    
    pygame.display.flip()

# ----- Cost Map & Path Planning (A*) Functions -----
def world_to_grid(point, cell_size):
    """Converts world coordinates (meters) to grid indices."""
    return (int(point[1] / cell_size), int(point[0] / cell_size))

def grid_to_world(grid_coord, cell_size):
    """Converts grid indices to world coordinates (meters), centered in the cell."""
    return ((grid_coord[1] + 0.5) * cell_size, (grid_coord[0] + 0.5) * cell_size)

def get_neighbors(node, grid):
    """Returns valid 8-connected neighbors for a grid node."""
    rows, cols = grid.shape
    neighbors = []
    for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
        r, c = node[0] + dr, node[1] + dc
        if 0 <= r < rows and 0 <= c < cols and grid[r, c] == 0:
            neighbors.append((r, c))
    return neighbors

def heuristic(a, b):
    """Euclidean distance heuristic."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def a_star(grid, start, goal):
    """Performs A* search on a 2D grid."""
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return [grid_to_world(p, CELL_SIZE) for p in reversed(path)]
        
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + (math.sqrt(2) if current[0] != neighbor[0] and current[1] != neighbor[1] else 1)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def plan_path(robot_pos, target_pos):
    """Generates a path using A* from robot_pos to target_pos."""
    grid_size = int(ROOM_SIZE / CELL_SIZE)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    # Mark room boundaries as obstacles.
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    start = world_to_grid(robot_pos, CELL_SIZE)
    goal = world_to_grid(target_pos, CELL_SIZE)
    return a_star(grid, start, goal)

# ----- LIDAR Scanning Thread -----
def scanning_thread():
    """
    Continuously retrieves LIDAR scans, computes distances to the back and right walls,
    updates the global robot position, and (if scanning_active) accumulates scan points.
    """
    global latest_robot_pos, accumulated_points, scanning_active
    scan_count = 0
    for scan in lidar.iter_scans():
        d_back = get_wall_distance(scan, BACK_TARGET)
        d_right = get_wall_distance(scan, RIGHT_TARGET)
        if d_back is not None and d_right is not None:
            latest_robot_pos = (d_back, d_right)
            scan_count += 1
            # Debug: print update every 20 scans
            if scan_count % 20 == 0:
                print("Updated robot position:", latest_robot_pos)
            # Accumulate points only during the detection phase.
            if scanning_active:
                pts = get_global_points(latest_robot_pos, scan)
                if pts.size > 0:
                    accumulated_points.extend(pts.tolist())
        # Continue updating latest_robot_pos continuously.

# ----- Main Execution -----
def main():
    global scanning_active, latest_robot_pos

    # Start the LIDAR scanning thread.
    thread = threading.Thread(target=scanning_thread, daemon=True)
    thread.start()

    print("Starting detection phase for", DETECTION_DURATION, "seconds...")
    time.sleep(DETECTION_DURATION)
    scanning_active = False  # End accumulation after detection phase

    if latest_robot_pos is None:
        latest_robot_pos = (ROOM_SIZE / 2, ROOM_SIZE / 2)
        print("No valid robot position detected during detection phase. Defaulting to:", latest_robot_pos)
    else:
        print("Final robot position after detection phase:", latest_robot_pos)

    # Process accumulated points to detect objects.
    filtered_points = filter_interior_points(np.array(accumulated_points), ROOM_SIZE, WALL_MARGIN)
    objects = cluster_objects(filtered_points)
    print("Detected objects (cluster centroids):", objects)
    
    # Plan a path to the first detected object (if any).
    path = None
    if objects:
        path = plan_path(latest_robot_pos, objects[0])
        print("Planned path:", path)
    else:
        print("No objects detected.")
    
    # Visualization loop: update display continuously using the latest robot position.
    running = True
    while running:
        draw_map(latest_robot_pos, objects, path)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        pygame.time.wait(50)
    
    lidar.stop()
    lidar.disconnect()
    pygame.quit()

if __name__ == "__main__":
    main()
