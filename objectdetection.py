import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import random

class LidarSimulator:
    """Simulates a LIDAR sensor for testing without hardware"""
    def __init__(self, room_walls, noise_factor=0.02):
        self.room_walls = room_walls  # List of [(x1,y1), (x2,y2)] wall segments
        self.noise_factor = noise_factor
        
    def scan(self, sensor_pos, num_angles=360):
        """Simulates a LIDAR scan from the given position"""
        points = []
        angles = np.linspace(0, 2*np.pi, num_angles)
        
        for angle in angles:
            # Send a ray from sensor position
            ray_dir = (math.cos(angle), math.sin(angle))
            closest_dist = float('inf')
            
            # Check intersection with each wall
            for wall in self.room_walls:
                dist = self._ray_segment_intersection(sensor_pos, ray_dir, wall)
                if dist and dist < closest_dist:
                    closest_dist = dist
            
            # If hit a wall, add the point with some noise
            if closest_dist < float('inf'):
                noise = random.uniform(-self.noise_factor, self.noise_factor)
                adjusted_dist = closest_dist * (1 + noise)
                point_x = sensor_pos[0] + ray_dir[0] * adjusted_dist
                point_y = sensor_pos[1] + ray_dir[1] * adjusted_dist
                points.append((point_x, point_y))
        
        return points
    
    def _ray_segment_intersection(self, ray_origin, ray_dir, segment):
        """Calculate intersection between a ray and a line segment"""
        p1, p2 = segment
        v1 = (ray_origin[0] - p1[0], ray_origin[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        v3 = (-ray_dir[1], ray_dir[0])
        
        dot = v2[0]*v3[0] + v2[1]*v3[1]
        if abs(dot) < 1e-6:  # Parallel
            return None
        
        t1 = (v2[0]*v1[1] - v2[1]*v1[0]) / dot
        t2 = (v1[0]*v3[0] + v1[1]*v3[1]) / dot
        
        if t1 >= 0 and 0 <= t2 <= 1:
            return t1
        return None

class RoomMapper:
    def __init__(self, initial_position=(0, 0), map_size=10, resolution=0.05):
        self.position = initial_position
        self.orientation = 0  # radians
        self.path = [initial_position]
        
        # Occupancy grid for mapping
        self.grid_size = int(map_size / resolution)
        self.resolution = resolution
        self.occupancy_grid = np.zeros((self.grid_size, self.grid_size))
        self.grid_origin = (self.grid_size//2, self.grid_size//2)  # Center of the grid
        
        # For visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.tight_layout()
        
        # For wall extraction
        self.wall_points = []
        self.detected_walls = []
    
    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices"""
        grid_x = int(self.grid_origin[0] + x / self.resolution)
        grid_y = int(self.grid_origin[1] + y / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid indices to world coordinates"""
        world_x = (grid_x - self.grid_origin[0]) * self.resolution
        world_y = (grid_y - self.grid_origin[1]) * self.resolution
        return world_x, world_y
    
    def update_map(self, scan_points):
        """Update occupancy grid based on LIDAR scan points"""
        # Decay existing map slightly (forgetting factor)
        self.occupancy_grid *= 0.99
        
        # Add new scan points
        for point in scan_points:
            grid_x, grid_y = self.world_to_grid(point[0], point[1])
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.occupancy_grid[grid_y, grid_x] = min(1.0, self.occupancy_grid[grid_y, grid_x] + 0.2)
                self.wall_points.append(point)
    
    def move(self, delta_x, delta_y, delta_theta=0):
        """Move the robot by the specified amounts"""
        # Update orientation
        self.orientation += delta_theta
        self.orientation %= 2 * np.pi
        
        # Transform movement by orientation
        rotated_dx = delta_x * math.cos(self.orientation) - delta_y * math.sin(self.orientation)
        rotated_dy = delta_x * math.sin(self.orientation) + delta_y * math.cos(self.orientation)
        
        # Update position
        self.position = (self.position[0] + rotated_dx, self.position[1] + rotated_dy)
        self.path.append(self.position)
        
        return self.position
    
    def extract_walls(self, min_points=20, max_distance=0.2):
        """Extract wall segments using RANSAC-inspired approach"""
        if len(self.wall_points) < min_points:
            return []
        
        walls = []
        points = np.array(self.wall_points)
        remaining_points = points.copy()
        
        # Attempt to find walls until we can't find more
        while len(remaining_points) > min_points:
            # Randomly select two points to form a line hypothesis
            idx1, idx2 = np.random.choice(len(remaining_points), 2, replace=False)
            p1, p2 = remaining_points[idx1], remaining_points[idx2]
            
            # Skip if points are too close
            if np.linalg.norm(p1 - p2) < 0.5:
                continue
                
            # Calculate line equation: ax + by + c = 0
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            norm = math.sqrt(a*a + b*b)
            
            # Find points that fit this line
            distances = np.abs(a*remaining_points[:,0] + b*remaining_points[:,1] + c) / norm
            inliers = remaining_points[distances < max_distance]
            
            # If we have enough inliers, consider this a wall
            if len(inliers) > min_points:
                # Fit a better line to all inliers using least squares
                x = inliers[:, 0]
                y = inliers[:, 1]
                
                if np.std(x) > np.std(y):  # More horizontal variation
                    coeffs = np.polyfit(x, y, 1)
                    # Find the endpoints by taking min and max x
                    x_min, x_max = np.min(x), np.max(x)
                    y_min = coeffs[0] * x_min + coeffs[1]
                    y_max = coeffs[0] * x_max + coeffs[1]
                    wall = [(x_min, y_min), (x_max, y_max)]
                else:  # More vertical variation
                    coeffs = np.polyfit(y, x, 1)
                    # Find the endpoints by taking min and max y
                    y_min, y_max = np.min(y), np.max(y)
                    x_min = coeffs[0] * y_min + coeffs[1]
                    x_max = coeffs[0] * y_max + coeffs[1]
                    wall = [(x_min, y_min), (x_max, y_max)]
                
                walls.append(wall)
                
                # Remove inliers from remaining points
                mask = np.ones(len(remaining_points), dtype=bool)
                for point in inliers:
                    mask = mask & ~np.all(remaining_points == point, axis=1)
                remaining_points = remaining_points[mask]
            
            # Prevent infinite loops if we can't find good walls
            if len(walls) > 10 or len(remaining_points) < min_points:
                break
        
        self.detected_walls = walls
        return walls
    
    def visualize(self, scan_points=None):
        """Visualize the current map, robot path, and detected walls"""
        self.ax.clear()
        
        # Show occupancy grid as a heatmap
        self.ax.imshow(self.occupancy_grid, cmap='binary', origin='lower', 
                       extent=[-self.grid_size/2*self.resolution, self.grid_size/2*self.resolution,
                              -self.grid_size/2*self.resolution, self.grid_size/2*self.resolution])
        
        # Show robot path
        path_x, path_y = zip(*self.path)
        self.ax.plot(path_x, path_y, 'b-', linewidth=1)
        
        # Show current position and orientation
        arrow_length = 0.3
        dx = arrow_length * math.cos(self.orientation)
        dy = arrow_length * math.sin(self.orientation)
        self.ax.arrow(self.position[0], self.position[1], dx, dy, 
                     head_width=0.1, head_length=0.15, fc='r', ec='r')
        
        # Show current scan points if provided
        if scan_points:
            scan_x, scan_y = zip(*scan_points)
            self.ax.scatter(scan_x, scan_y, c='g', s=10, alpha=0.5)
        
        # Show detected walls
        for wall in self.detected_walls:
            x_values = [wall[0][0], wall[1][0]]
            y_values = [wall[0][1], wall[1][1]]
            self.ax.plot(x_values, y_values, 'r-', linewidth=2)
        
        self.ax.set_title('Room Mapping and Wall Detection')
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        
        plt.draw()
        plt.pause(0.01)

def main():
    # Define a rectangular room with walls
    room_walls = [
        [(0, 0), (5, 0)],     # Bottom wall
        [(5, 0), (5, 4)],     # Right wall
        [(5, 4), (0, 4)],     # Top wall
        [(0, 4), (0, 0)],     # Left wall
        [(2, 2), (4, 2)]      # Interior wall segment
    ]
    
    # Initialize simulator and mapper
    lidar = LidarSimulator(room_walls)
    mapper = RoomMapper(initial_position=(1, 1), map_size=8, resolution=0.05)
    
    # Animation function for real-time updates
    def update_frame(frame):
        # Move in a complex pattern to explore the room
        if frame % 20 == 0:
            # Change direction periodically
            theta_change = np.random.uniform(-0.3, 0.3)
            mapper.move(0.2, 0, theta_change)
        else:
            mapper.move(0.1, 0, 0)  # Move forward
        
        # Get simulated LIDAR scan from current position
        scan_points = lidar.scan(mapper.position)
        
        # Update the map
        mapper.update_map(scan_points)
        
        # Periodically extract walls
        if frame % 10 == 0:
            mapper.extract_walls()
        
        # Visualize
        mapper.visualize(scan_points)
        
        return []

    # Start animation
    print("Starting room mapping simulation...")
    anim = FuncAnimation(mapper.fig, update_frame, frames=200, interval=100, blit=True)
    plt.show()
    
    print(f"Mapping complete. Detected {len(mapper.detected_walls)} walls.")
    
    # Final visualization with extracted walls
    mapper.extract_walls(min_points=15, max_distance=0.15)  # More precise final extraction
    mapper.visualize()
    plt.savefig('room_map_final.png')
    plt.show()

if __name__ == "__main__":
    main()
