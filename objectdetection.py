import serial
import time
import math
import pygame
import numpy as np

# LIDAR Serial Port Configuration
LIDAR_PORT = "/dev/ttyUSB0"
BAUDRATE = 460800
TIMEOUT = 1

# Pygame Configuration
WINDOW_SIZE = (800, 800)
SCALE_FACTOR = 50  # Scale for 8x8 meter area
BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (50, 50, 50)
LIDAR_COLOR = (255, 0, 0)
POINT_COLOR = (255, 255, 0)
LIDAR_POSITION_COLOR = (0, 255, 255)

class Lidar360:
    def __init__(self, port=LIDAR_PORT, baudrate=BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_scanning = False
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("LIDAR Area Mapping")
        
        self.font = pygame.font.Font(None, 36)
        self.raw_points = []
        self.lidar_position = None
        self.reference_corner = None  # Reference (bottom-left corner)
        self.reference_detected = False  # Flag to track if reference is established
    
    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=TIMEOUT)
            if self.serial.is_open:
                print(f"Connected to LIDAR on {self.port}")
                return True
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            return False

    def detect_bottom_left_corner(self, points):
        """Find the bottom-left corner based on perpendicular walls."""
        if len(points) < 50:
            return None
        
        points = np.array(points)
        # Find points that likely represent walls
        # We can improve this by finding the most dense cluster of points
        # at the extremes of the scan area
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
        
        # Find points within a small distance of the corner
        corner_candidate_indices = np.where(
            (points[:, 0] < min_x + 0.2) & 
            (points[:, 1] < min_y + 0.2)
        )[0]
        
        if len(corner_candidate_indices) > 5:
            corner_candidates = points[corner_candidate_indices]
            # Use the average of these points as a more robust corner estimate
            corner = np.mean(corner_candidates, axis=0)
            return (corner[0], corner[1])
        
        return (min_x, min_y)
    
    def world_to_screen(self, x, y):
        """Convert world coordinates (relative to bottom-left) to screen coordinates."""
        if self.reference_corner:
            x -= self.reference_corner[0]
            y -= self.reference_corner[1]
        screen_x = int(x * SCALE_FACTOR)
        screen_y = int(WINDOW_SIZE[1] - (y * SCALE_FACTOR))  # Flip y-axis
        return (screen_x, screen_y)
    
    def draw_grid(self):
        """Draw a grid from the reference point."""
        for i in range(9):
            x = i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_SIZE[1]))
            y = WINDOW_SIZE[1] - i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y))
    
    def draw_lidar_position(self):
        if self.lidar_position and self.reference_corner:
            screen_pos = self.world_to_screen(*self.lidar_position)
            pygame.draw.circle(self.screen, LIDAR_POSITION_COLOR, screen_pos, 8)
            
            # Display position relative to reference point
            rel_x = self.lidar_position[0] - self.reference_corner[0]
            rel_y = self.lidar_position[1] - self.reference_corner[1]
            text = f"LIDAR Position: ({rel_x:.2f}m, {rel_y:.2f}m) from reference"
            self.screen.blit(self.font.render(text, True, LIDAR_POSITION_COLOR), (10, 10))
            
            # Also display the reference corner
            ref_text = f"Reference Corner: ({self.reference_corner[0]:.2f}m, {self.reference_corner[1]:.2f}m)"
            self.screen.blit(self.font.render(ref_text, True, GRID_COLOR), (10, 50))
    
    def start_scan(self):
        self.send_command(b'\xA5\x20')
        self.is_scanning = True

    def stop_scan(self):
        self.send_command(b'\xA5\x25')
        self.is_scanning = False
        self.serial.close()
        pygame.quit()

    def send_command(self, command):
        self.serial.write(command)
        time.sleep(0.1)

    def read_and_display(self):
        points = []
        update_counter = 0
        while self.is_scanning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            data = self.serial.read(5)
            if len(data) < 5:
                continue
            
            quality = data[0] >> 2
            angle = ((data[1] | ((data[2] & 0x7F) << 8)) / 64.0)
            distance = (data[3] | (data[4] << 8)) / 4.0  # mm
            
            if quality > 0 and 0 < distance < 6000:
                x = distance * math.cos(math.radians(angle)) / 1000
                y = distance * math.sin(math.radians(angle)) / 1000
                points.append((x, y))
                
                update_counter += 1
                if update_counter >= 10:
                    update_counter = 0
                    
                    # First establish the reference corner if not already done
                    if not self.reference_detected and len(points) > 100:
                        self.reference_corner = self.detect_bottom_left_corner(points)
                        if self.reference_corner:
                            self.reference_detected = True
                            print(f"Reference corner detected at: {self.reference_corner}")
                    
                    # Calculate LIDAR position only after reference is established
                    if self.reference_detected:
                        self.lidar_position = self.calculate_lidar_position(points)
                
                self.screen.fill(BACKGROUND_COLOR)
                self.draw_grid()
                
                # Draw reference corner if detected
                if self.reference_corner:
                    ref_screen_pos = self.world_to_screen(*self.reference_corner)
                    pygame.draw.circle(self.screen, (0, 255, 0), ref_screen_pos, 8)
                
                for px, py in points:
                    pygame.draw.circle(self.screen, POINT_COLOR, self.world_to_screen(px, py), 2)
                
                self.draw_lidar_position()
                pygame.display.flip()
                
                if len(points) > 1000:
                    points = points[-1000:]
    
    def calculate_lidar_position(self, points):
        """Calculate LIDAR position as absolute coordinates in the room."""
        if len(points) < 100 or not self.reference_corner:
            return None
            
        # Use the current points to estimate LIDAR position
        # For improved accuracy, we could implement a more sophisticated algorithm here
        points_array = np.array(points[-100:])
        
        # Calculate average distance in all directions
        distances = np.sqrt(points_array[:, 0]**2 + points_array[:, 1]**2)
        avg_distance = np.mean(distances)
        
        # For simplicity, we'll just use (0,0) as the LIDAR's physical position
        # and let all measurements be relative to the reference corner
        return (0, 0)  # This means the LIDAR is at the origin of its own coordinate system
    
if __name__ == "__main__":
    lidar = Lidar360()
    try:
        if lidar.connect():
            lidar.start_scan()
            lidar.read_and_display()
    except KeyboardInterrupt:
        print("\nStopping scan...")
        lidar.stop_scan()
        print("LIDAR disconnected.")
