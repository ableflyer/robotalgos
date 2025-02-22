import pygame
import math
import numpy as np
from rplidar import RPLidar
from sklearn.cluster import DBSCAN
import time
import heapq

# ----- Configuration -----
PORT_NAME = '/dev/ttyUSB0'     # Change to your LIDAR port (e.g., 'COM3' on Windows)
BAUDRATE = 460800              # RPLidar C1 baudrate
ROOM_SIZE = 8.0                # Room dimensions: 8x8 meters
SCALE = 100                    # Scale for visualization: 100 pixels per meter
WALL_MARGIN = 0.2              # Margin (in meters) to consider points "touching" a wall
UPSIDE_DOWN = True             # Set True if the LIDAR is mounted upside down
CELL_SIZE = 0.1                # Resolution for the cost-map grid (in meters)
DETECTION_DURATION = 5.0       # Duration (in seconds) to accumulate scan points

# ----- Initialize RPLidar -----
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# ----- Initialize Pygame -----
WINDOW_SIZE = (int(ROOM_SIZE * SCALE), int(ROOM_SIZE * SCALE))
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Mapping, Localization & Path Planning (Upside Down)")

# Colors for visualization
BACKGROUND_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)       # Walls drawn in red
ROBOT_COLOR = (0, 255, 0)      # Robot position drawn in green
OBJECT_COLOR = (0, 0, 255)     # Detected objects drawn in blue
PATH_COLOR = (255, 255, 0)     # Planned path drawn in yellow

def get_wall_distance(scan, target_angle, angle_tolerance=10):
    """
    Computes the average distance (in meters) for LIDAR measurements
    within target_angle ± angle_tolerance.
    """
    distances = []
    for quality, angle, distance in scan:
        if abs(angle - target_angle) < angle_tolerance:
            if distance > 0:
                distances.append(distance / 1000.0)  # Convert mm to meters
    if distances:
        return np.mean(distances)
    return None

def get_global_points(robot_pos, scan):
    """
    Converts LIDAR scan measurements (in the sensor's polar frame) into global coordinates.
    Adjusts the conversion if the sensor is mounted upside down.
    """
    points = []
    for quality, angle, distance in scan:
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
    """
    Remove points that are within 'margin' distance of any wall.
    Walls are assumed to be at x=0, y=0, x=room_size, and y=room_size.
    """
    filtered = []
    for x, y in points:
        if margin < x < (room_size - margin) and margin < y < (room_size - margin):
            filtered.append((x, y))
    return np.array(filtered) if filtered else np.empty((0, 2))

def cluster_objects(points, eps=0.2, min_samples=3):
    """
    Uses DBSCAN to cluster the given points and returns the centroids
    of clusters that are considered as objects.
    """
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    object_positions = []
    for label in set(labels):
        if label == -1:  # Noise
            continue
        cluster_points = points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        object_positions.append(centroid)
    return object_positions

def draw_map(robot_pos, objects, path=None):
    """
    Draw the room boundaries, the robot's estimated position, detected objects,
    and (if available) the planned path.
    """
    screen.fill(BACKGROUND_COLOR)
    # Draw room boundary
    rect = pygame.Rect(0, 0, ROOM_SIZE * SCALE, ROOM_SIZE * SCALE)
    pygame.draw.rect(screen, WALL_COLOR, rect, 2)
    
    # Draw robot as a green circle
    if robot_pos:
        rx, ry = robot_pos
        robot_screen = (int(rx * SCALE), int(WINDOW_SIZE[1] - ry * SCALE))
        pygame.draw.circle(screen, ROBOT_COLOR, robot_screen, 5)
    
    # Draw each localized object as a blue circle
    for obj in objects:
        ox, oy = obj
        obj_screen = (int(ox * SCALE), int(WINDOW_SIZE[1] - oy * SCALE))
        pygame.draw.circle(screen, OBJECT_COLOR, obj_screen, 4)
    
    # Draw the planned path (if any) as yellow lines
    if path:
        path_points = []
        for (x, y) in path:
            screen_x = int(x * SCALE)
            screen_y = int(WINDOW_SIZE[1] - y * SCALE)
            path_points.append((screen_x, screen_y))
        if len(path_points) > 1:
            pygame.draw.lines(screen, PATH_COLOR, False, path_points, 2)
    
    pygame.display.flip()

# ----- Helper functions for A* pathfinding -----
def world_to_grid(point, cell_size):
    """Convert world (meter) coordinates to grid indices (row, col)."""
    x, y = point
    col = int(x / cell_size)
    row = int(y / cell_size)
    return (row, col)

def grid_to_world(grid_coord, cell_size):
    """Convert grid indices (row, col) to world coordinates (meters), centered in the cell."""
    row, col = grid_coord
    x = (col + 0.5) * cell_size
    y = (row + 0.5) * cell_size
    return (x, y)

def get_neighbors(node, grid):
    """Return valid 8-connected neighbors of the given node in the grid."""
    neighbors = []
    rows, cols = grid.shape
    row, col = node
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols:
                # Check if the cell is free (0 means free, 1 means obstacle)
                if grid[r, c] == 0:
                    neighbors.append((r, c))
    return neighbors

def move_cost(current, neighbor):
    """Return the cost to move from current to neighbor (diagonal moves cost sqrt2)."""
    if current[0] != neighbor[0] and current[1] != neighbor[1]:
        return math.sqrt(2)
    return 1

def heuristic(a, b):
    """Euclidean distance heuristic."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, current):
    """Reconstruct path from start to current node."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def a_star(grid, start, goal):
    """A* pathfinding on a 2D grid."""
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            return reconstruct_path(came_from, current)
        
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + move_cost(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def plan_path(robot_pos, target_pos, cell_size):
    """
    Create a simple cost map (grid) for the room and use A* to plan a path
    from robot_pos to target_pos.
    """
    grid_size = int(ROOM_SIZE / cell_size)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    # Mark boundaries as obstacles.
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    start = world_to_grid(robot_pos, cell_size)
    goal = world_to_grid(target_pos, cell_size)
    
    path_grid = a_star(grid, start, goal)
    if path_grid is None:
        return None
    # Convert grid coordinates back to world coordinates.
    path_world = [grid_to_world(node, cell_size) for node in path_grid]
    return path_world

# ----- Main loop -----
def main():
    print("Mapping, Localization & Path Planning started.")
    print(f"Accumulating scan points for {DETECTION_DURATION} seconds...")
    accumulated_interior_points = []
    robot_pos = None

    # Choose target angles based on mounting orientation.
    if UPSIDE_DOWN:
        back_target = 180   # Back wall remains at 180°
        right_target = 90   # Right wall (from global perspective) now detected at 90° in sensor frame
    else:
        back_target = 180
        right_target = 270
    
    detection_start = time.time()
    # --- Detection Phase ---
    while time.time() - detection_start < DETECTION_DURATION:
        for scan in lidar.iter_scans():
            d_back = get_wall_distance(scan, back_target, angle_tolerance=10)
            d_right = get_wall_distance(scan, right_target, angle_tolerance=10)
            if d_back is not None and d_right is not None:
                # Estimate robot position from the current scan.
                robot_pos = (d_back, d_right)
                # Convert the scan to global coordinates.
                global_points = get_global_points(robot_pos, scan)
                # Filter out points near walls.
                interior_points = filter_interior_points(global_points, ROOM_SIZE, WALL_MARGIN)
                if len(interior_points) > 0:
                    accumulated_interior_points.extend(interior_points.tolist())
            # Check if detection duration is reached.
            if time.time() - detection_start >= DETECTION_DURATION:
                break

    # Convert accumulated points to a NumPy array.
    if accumulated_interior_points:
        all_points = np.array(accumulated_interior_points)
    else:
        all_points = np.empty((0, 2))
    
    # Cluster the accumulated points to detect objects.
    object_list = cluster_objects(all_points)
    print("Detected objects (cluster centroids):", object_list)
    
    # --- Path Planning Phase ---
    planned_path = None
    if object_list and robot_pos is not None:
        target = object_list[0]
        planned_path = plan_path(robot_pos, target, CELL_SIZE)
        print("Planning path to object at:", target)
    else:
        print("No valid object detected or robot position unavailable.")
    
    # --- Visualization Loop ---
    running = True
    while running:
        draw_map(robot_pos, object_list, planned_path)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    # Cleanup on exit.
    lidar.stop()
    lidar.disconnect()
    pygame.quit()

if __name__ == "__main__":
    main()
