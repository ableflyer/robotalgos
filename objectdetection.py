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
        self.reference_corner = None  # New reference (bottom-left corner)
    
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
        min_x = np.min(points[:, 0])
        min_y = np.min(points[:, 1])
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
        """Draw a grid from the new reference point."""
        for i in range(9):
            x = i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_SIZE[1]))
            y = WINDOW_SIZE[1] - i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y))
    
    def draw_lidar_position(self):
        if self.lidar_position:
            screen_pos = self.world_to_screen(*self.lidar_position)
            pygame.draw.circle(self.screen, LIDAR_POSITION_COLOR, screen_pos, 8)
            text = f"LIDAR: ({self.lidar_position[0]:.2f}m, {self.lidar_position[1]:.2f}m)"
            self.screen.blit(self.font.render(text, True, LIDAR_POSITION_COLOR), (10, 10))
    
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
                    if self.reference_corner is None:
                        self.reference_corner = self.detect_bottom_left_corner(points)
                    self.lidar_position = self.detect_lidar_position(points)
                
                self.screen.fill(BACKGROUND_COLOR)
                self.draw_grid()
                
                for px, py in points:
                    pygame.draw.circle(self.screen, POINT_COLOR, self.world_to_screen(px, py), 2)
                
                self.draw_lidar_position()
                pygame.display.flip()
                
                if len(points) > 1000:
                    points = points[-1000:]
    
    def detect_lidar_position(self, points):
        if len(points) < 50:
            return None
        points_array = np.array(points[-50:])
        centroid = np.mean(points_array, axis=0)
        if self.reference_corner:
            return (centroid[0] - self.reference_corner[0], centroid[1] - self.reference_corner[1])
        return None
    
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
# change 1
