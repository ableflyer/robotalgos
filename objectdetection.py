# a comment just to check that it's updated
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
# For a normally mounted LIDAR, one might use 180° for the back wall and 270° for the right wall.
# When mounted upside down, the right wall is detected at 90°.
BACK_TARGET = 180
RIGHT_TARGET = 90 if UPSIDE_DOWN else 270

# ----- Global Variables -----
latest_robot_pos = None      # Continuously updated robot position (in meters)
accumulated_points = []      # Accumulated global scan points for object detection
scanning_active = True       # Flag to control accumulation during the detection phase

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
    Computes the average distance (in meters) for LIDAR measurements
    within target_angle ± angle_tolerance.
    """
    distances = [d / 1000.0 for _, angle, d in scan
                 if abs(angle - target_angle) < angle_tolerance and d > 0]
    return np.mean(distances) if distances else None

def get_global_points(robot_pos, scan):
    """
    Converts LIDAR polar coordinates (angle, distance) into global Cartesian coordinates.
    Adjusts the conversion if the LIDAR is mounted upside down.
    """
    points = []
    for _, angle, distance in scan:
        if distance > 0:
            d = distance / 1000.0  # Convert mm to meters
            rad = math.radians(angle)
            x_local = d * math.cos(rad)
            y_local = -d * math.sin(rad) if UPSIDE_DOWN else d * math.sin(rad)
            global_x = robot_pos[0] + x_local
            global_y = robot_pos[1] + y_local
            points.append((global_x, global_y))
    return np.array(points)

def filter_interior_points(points, room_size, margin):
    """Removes points too close to the room walls (within margin)."""
    return np.array([p for p in points if margin < p[0] < (room_size - margin) and margin < p[1] < (room_size - margin)])

def cluster_objects(points, eps=0.2, min_samples=3):
    """
    Uses DBSCAN clustering to group accumulated points and returns the centroids
    of clusters (which are considered as detected objects).
    """
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return [np.mean(points[labels == label], axis=0) for label in set(labels) if label != -1]

# ----- Drawing Function -----
def draw_map(robot_pos, objects, path=None):
    """
    Draws the room boundaries, the robot's estimated position, detected objects,
    and the planned path (if available).
    Coordinates are converted to screen space (with y=0 at the bottom).
    """
    screen.fill(BACKGROUND_COLOR)
    
    # Draw room boundary
    pygame.draw.rect(screen, WALL_COLOR, pygame.Rect(0, 0, ROOM_SIZE * SCALE, ROOM_SIZE * SCALE), 2)
    
    # Draw robot position
    if robot_pos:
        rx, ry = robot_pos
        screen_x = int(rx * SCALE)
        screen_y = int(WINDOW_SIZE[1] - ry * SCALE)
        pygame.draw.circle(screen, ROBOT_COLOR, (screen_x, screen_y), 5)
    
    # Draw detected objects
    for obj in objects:
        ox, oy = obj
        obj_screen = (int(ox * SCALE), int(WINDOW_SIZE[1] - oy * SCALE))
        pygame.draw.circle(screen, OBJECT_COLOR, obj_screen, 4)
    
    # Draw planned path if available
    if path and len(path) > 1:
        path_points = [(int(x * SCALE), int(WINDOW_SIZE[1] - y * SCALE)) for x, y in path]
        pygame.draw.lines(screen, PATH_COLOR, False, path_points, 2)
    
    pygame.display.flip()

# ----- Cost Map & Path Planning Functions (A*) -----
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
    """Generates a path from robot_pos to target_pos using A* on a cost-map grid."""
    grid_size = int(ROOM_SIZE / CELL_SIZE)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    # Mark room boundaries as obstacles.
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
    start = world_to_grid(robot_pos, CELL_SIZE)
    goal = world_to_grid(target_pos, CELL_SIZE)
    return a_star(grid, start, goal)

# ----- LIDAR Scanning Thread -----
def scanning_thread():
    
