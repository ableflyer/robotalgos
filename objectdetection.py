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
SCALE_FACTOR = 50  # Reduced to better show 8x8 meter area
BACKGROUND_COLOR = (0, 0, 0)
GRID_COLOR = (50, 50, 50)
LIDAR_COLOR = (255, 0, 0)
AREA_COLOR = (0, 255, 0, 100)
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
        
        # Initialize font for text display
        self.font = pygame.font.Font(None, 36)
        
        # Mapping state
        self.area_center = (0, 0)  # Center of the 8x8 meter area
        self.lidar_position = None
        self.raw_points = []  # Store raw LIDAR points
        
    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=TIMEOUT)
            if self.serial.is_open:
                print(f"Connected to LIDAR on {self.port}")
                return True
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            return False

    def detect_lidar_position(self, points):
        """
        Detect LIDAR position using wall points.
        Simplified version that uses clustering of recent points.
        """
        if len(points) < 50:  # Need minimum points for reliable detection
            return None
            
        # Convert points to numpy array for easier processing
        points_array = np.array(points[-50:])  # Use last 50 points
        
        # Calculate centroid of recent points
        centroid = np.mean(points_array, axis=0)
        x, y = centroid
        
        # Check if position is within the 8x8 meter area
        if self.is_point_in_area(x, y):
            return (x, y)
        return None

    def world_to_screen(self, x, y):
        """Convert world coordinates (meters) to screen coordinates (pixels)."""
        screen_x = int(WINDOW_SIZE[0]/2 + x * SCALE_FACTOR)
        screen_y = int(WINDOW_SIZE[1]/2 - y * SCALE_FACTOR)
        return (screen_x, screen_y)

    def draw_grid(self):
        """Draw a grid on the screen."""
        # Draw center lines
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (WINDOW_SIZE[0]/2, 0), 
                        (WINDOW_SIZE[0]/2, WINDOW_SIZE[1]), 2)
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (0, WINDOW_SIZE[1]/2), 
                        (WINDOW_SIZE[0], WINDOW_SIZE[1]/2), 2)
        
        # Draw grid lines
        for i in range(-8, 9):
            # Vertical lines
            x = WINDOW_SIZE[0]/2 + i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (x, 0), (x, WINDOW_SIZE[1]))
            
            # Horizontal lines
            y = WINDOW_SIZE[1]/2 + i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, 
                           (0, y), (WINDOW_SIZE[0], y))

    def draw_area(self):
        """Draw the centered 8x8 meter area."""
        # Calculate corner points for 8x8 meter area
        half_size = 4  # Half of 8 meters
        vertices = [
            self.world_to_screen(-half_size, half_size),   # Top left
            self.world_to_screen(half_size, half_size),    # Top right
            self.world_to_screen(half_size, -half_size),   # Bottom right
            self.world_to_screen(-half_size, -half_size)   # Bottom left
        ]
        
        # Draw area outline
        pygame.draw.polygon(self.screen, AREA_COLOR, vertices, 2)
        
        # Draw measurements
        text_color = (200, 200, 200)
        # Draw "8m" labels
        text = self.font.render("8m", True, text_color)
        self.screen.blit(text, (WINDOW_SIZE[0]/2 + 5, WINDOW_SIZE[1]/2 - 4 * SCALE_FACTOR))
        # Rotate text for vertical measurement
        rotated_text = pygame.transform.rotate(text, 90)
        self.screen.blit(rotated_text, (WINDOW_SIZE[0]/2 - 4 * SCALE_FACTOR - 20, WINDOW_SIZE[1]/2 + 5))

    def draw_lidar_position(self):
        """Draw the LIDAR position if detected."""
        if self.lidar_position:
            x, y = self.lidar_position
            screen_pos = self.world_to_screen(x, y)
            
            # Draw LIDAR position marker
            pygame.draw.circle(self.screen, LIDAR_POSITION_COLOR, screen_pos, 8)
            
            # Draw crosshair
            size = 15
            pygame.draw.line(self.screen, LIDAR_POSITION_COLOR,
                           (screen_pos[0] - size, screen_pos[1]),
                           (screen_pos[0] + size, screen_pos[1]), 2)
            pygame.draw.line(self.screen, LIDAR_POSITION_COLOR,
                           (screen_pos[0], screen_pos[1] - size),
                           (screen_pos[0], screen_pos[1] + size), 2)
            
            # Display coordinates
            text = f"LIDAR: ({x:.2f}m, {y:.2f}m)"
            text_surface = self.font.render(text, True, LIDAR_POSITION_COLOR)
            self.screen.blit(text_surface, (10, 10))

    def is_point_in_area(self, x, y):
        """Check if a point is within the 8x8 meter area."""
        return abs(x) <= 4 and abs(y) <= 4  # Using half-size (4m) from center

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
        """Read LIDAR data and display in Pygame window."""
        points = []
        update_counter = 0
        
        while self.is_scanning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            # Read LIDAR data
            data = self.serial.read(5)
            if len(data) < 5:
                continue
                
            # Decode LIDAR packet
            quality = data[0] >> 2
            angle = ((data[1] | ((data[2] & 0x7F) << 8)) / 64.0)
            distance = (data[3] | (data[4] << 8)) / 4.0  # mm
            
            if quality > 0 and 0 < distance < 6000:
                # Convert polar to cartesian coordinates (in meters)
                x = distance * math.cos(math.radians(angle)) / 1000
                y = distance * math.sin(math.radians(angle)) / 1000
                points.append((x, y))
                
                # Update counter for periodic processing
                update_counter += 1
                
                if update_counter >= 10:  # Process every 10 points
                    update_counter = 0
                    self.lidar_position = self.detect_lidar_position(points)
                
                # Update display
                self.screen.fill(BACKGROUND_COLOR)
                self.draw_grid()
                self.draw_area()
                
                # Draw LIDAR points
                for px, py in points:
                    screen_pos = self.world_to_screen(px, py)
                    if self.is_point_in_area(px, py):
                        pygame.draw.circle(self.screen, POINT_COLOR, screen_pos, 2)
                    else:
                        pygame.draw.circle(self.screen, LIDAR_COLOR, screen_pos, 2)
                
                # Draw LIDAR position
                self.draw_lidar_position()
                
                pygame.display.flip()
                
                # Limit points list size
                if len(points) > 1000:
                    points = points[-1000:]

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
