import pygame
import math
import numpy as np
from rplidar import RPLidar
from sklearn.cluster import DBSCAN

# ----- Configuration -----
PORT_NAME = '/dev/ttyUSB0'     # Change to your LIDAR port (e.g., 'COM3' on Windows)
BAUDRATE = 460800              # RPLidar C1 baudrate
ROOM_SIZE = 8.0                # Room dimensions: 8x8 meters
SCALE = 100                    # Scale for visualization: 100 pixels per meter
WALL_MARGIN = 0.2              # Margin (in meters) to consider points "touching" a wall
UPSIDE_DOWN = True             # Set True if the LIDAR is mounted upside down

# ----- Initialize RPLidar -----
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# ----- Initialize Pygame -----
WINDOW_SIZE = (int(ROOM_SIZE * SCALE), int(ROOM_SIZE * SCALE))
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Mapping & Localization Using Room Walls (Upside Down)")

# Colors for visualization
BACKGROUND_COLOR = (0, 0, 0)
WALL_COLOR = (255, 0, 0)       # Walls drawn in red
ROBOT_COLOR = (0, 255, 0)      # Robot position drawn in green
OBJECT_COLOR = (0, 0, 255)     # Detected objects drawn in blue

# Global list to store persistent object centroids (in global coordinates, in meters)
persistent_objects = []

def unpack_measurement(measurement):
    """
    Unpacks a measurement tuple from the LIDAR.
    Supports 4-tuple, 3-tuple, or 2-tuple formats.
    Returns: (quality, angle, distance, extra)
    """
    if len(measurement) == 4:
        return measurement
    elif len(measurement) == 3:
        quality, angle, distance = measurement
        return quality, angle, distance, None
    elif len(measurement) == 2:
        angle, distance = measurement
        return None, angle, distance, None
    else:
        raise ValueError("Unexpected measurement format: {}".format(measurement))

def get_wall_distance(scan, target_angle, angle_tolerance=10):
    """
    Computes the average distance (in meters) for LIDAR measurements
    within target_angle ± angle_tolerance.
    
    For example, if UPSIDE_DOWN is False, use target_angle=180 for the back wall
    and 270 for the right wall. If UPSIDE_DOWN is True, then we use 180 for the
    back wall and 90 for the wall that is now on the "right" (due to the flip).
    """
    distances = []
    for measurement in scan:
        quality, angle, distance, _ = unpack_measurement(measurement)
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
    for measurement in scan:
        quality, angle, distance, _ = unpack_measurement(measurement)
        if distance > 0:
            d = distance / 1000.0  # Convert mm to meters
            rad = math.radians(angle)
            x_local = d * math.cos(rad)
            # If upside down, flip the y component.
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

def update_persistent_objects(new_objects, threshold=0.3):
    """
    Updates the persistent_objects list with new objects that are not
    already detected within a certain threshold (in meters).
    """
    global persistent_objects
    for obj in new_objects:
        exists = False
        for existing in persistent_objects:
            if np.linalg.norm(np.array(obj) - np.array(existing)) < threshold:
                exists = True
                break
        if not exists:
            persistent_objects.append(obj)

def draw_map(robot_pos, objects):
    """
    Draw the room boundaries, the robot's estimated position, and squares
    representing localized objects. Each square is centered at the object's
    position and has side length 0.6 meters.
    """
    screen.fill(BACKGROUND_COLOR)
    # Draw room boundary
    rect = pygame.Rect(0, 0, ROOM_SIZE * SCALE, ROOM_SIZE * SCALE)
    pygame.draw.rect(screen, WALL_COLOR, rect, 2)
    
    # Draw robot as a green circle
    rx, ry = robot_pos
    robot_screen = (int(rx * SCALE), int(WINDOW_SIZE[1] - ry * SCALE))
    pygame.draw.circle(screen, ROBOT_COLOR, robot_screen, 5)
    
    # Draw each persistent object as a blue square
    square_side = 0.6 * SCALE  # 0.6 meters in pixels
    half_side = square_side / 2
    for obj in objects:
        ox, oy = obj
        center_x = int(ox * SCALE)
        center_y = int(WINDOW_SIZE[1] - oy * SCALE)
        square_rect = pygame.Rect(center_x - half_side, center_y - half_side,
                                  square_side, square_side)
        pygame.draw.rect(screen, OBJECT_COLOR, square_rect, 2)
    
    pygame.display.flip()

def main():
    print("Mapping & Localization started. Close the window to stop.")
    
    # For a moving LIDAR, we integrate the differences in the measured wall distances.
    global_robot_pos = None
    prev_measured_pos = None

    try:
        # Choose target angles based on mounting orientation.
        if UPSIDE_DOWN:
            back_target = 180   # Back wall remains at 180°
            right_target = 90   # Right wall (from global perspective) now at 90° in sensor frame
        else:
            back_target = 180
            right_target = 270
            
        while True:
            # Process one LIDAR scan
            for scan in lidar.iter_scans():
                # Estimate distances to walls from current scan:
                d_back = get_wall_distance(scan, back_target, angle_tolerance=10)
                d_right = get_wall_distance(scan, right_target, angle_tolerance=10)
                
                if d_back is not None and d_right is not None:
                    measured_pos = (d_back, d_right)
                    # For the first scan, initialize global position
                    if global_robot_pos is None:
                        global_robot_pos = measured_pos
                        prev_measured_pos = measured_pos
                    else:
                        # Compute the difference between current and previous measured positions.
                        delta = (measured_pos[0] - prev_measured_pos[0],
                                 measured_pos[1] - prev_measured_pos[1])
                        # Update global robot position by integrating the delta.
                        global_robot_pos = (global_robot_pos[0] + delta[0],
                                            global_robot_pos[1] + delta[1])
                        prev_measured_pos = measured_pos

                    print("Global Robot Position (meters):", global_robot_pos)
                    
                    # Convert the entire scan to global coordinates based on the updated robot position.
                    global_points = get_global_points(global_robot_pos, scan)
                    
                    # Filter out points that are too close to the walls.
                    interior_points = filter_interior_points(global_points, ROOM_SIZE, WALL_MARGIN)
                    
                    # Cluster the remaining points to detect objects.
                    new_objects = cluster_objects(interior_points)
                    
                    # Update persistent objects (only add new ones).
                    update_persistent_objects(new_objects, threshold=0.3)
                    
                    # Draw the room, the updated robot position, and persistent detected objects as squares.
                    draw_map(global_robot_pos, persistent_objects)
                
                # Handle Pygame events.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                # Process one scan per loop iteration.
                break
                
    except KeyboardInterrupt:
        print("Stopping mapping and localization.")
    finally:
        lidar.stop()
        lidar.disconnect()
        pygame.quit()

if __name__ == "__main__":
    main()
