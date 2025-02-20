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

        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("LIDAR Area Mapping")
        self.font = pygame.font.Font(None, 36)

        self.raw_points = []
        self.lidar_position = None
        self.reference_corner = None
        self.reference_detected = False

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
        if len(points) < 50:
            return None
        points = np.array(points)
        min_x, min_y = np.min(points[:, 0]), np.min(points[:, 1])
        corner_candidates = points[
            (points[:, 0] < min_x + 0.2) & (points[:, 1] < min_y + 0.2)
        ]
        return np.mean(corner_candidates, axis=0) if len(corner_candidates) > 5 else (min_x, min_y)

    def world_to_screen(self, x, y):
        if self.reference_corner:
            x -= self.reference_corner[0]
            y -= self.reference_corner[1]
        screen_x = int(x * SCALE_FACTOR)
        screen_y = int(WINDOW_SIZE[1] - (y * SCALE_FACTOR))  # Flip y-axis
        return screen_x, screen_y

    def draw_grid(self):
        for i in range(9):
            x, y = i * SCALE_FACTOR, WINDOW_SIZE[1] - i * SCALE_FACTOR
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, WINDOW_SIZE[1]))
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WINDOW_SIZE[0], y))

    def draw_lidar_position(self):
        if self.lidar_position and self.reference_corner:
            screen_pos = self.world_to_screen(*self.lidar_position)
            pygame.draw.circle(self.screen, LIDAR_POSITION_COLOR, screen_pos, 8)

            rel_x = self.lidar_position[0] - self.reference_corner[0]
            rel_y = self.lidar_position[1] - self.reference_corner[1]
            self.screen.blit(self.font.render(f"LIDAR Position: ({rel_x:.2f}m, {rel_y:.2f}m)", True, LIDAR_POSITION_COLOR), (10, 10))
            self.screen.blit(self.font.render(f"Reference Corner: ({self.reference_corner[0]:.2f}m, {self.reference_corner[1]:.2f}m)", True, GRID_COLOR), (10, 50))

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
        while self.is_scanning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            data = self.serial.read(5)
            if len(data) < 5:
                continue

            # Debug raw data
            print(f"Raw Data: {data}")

            # Verify checksum
            checksum = data[0] ^ data[1] ^ data[2] ^ data[3] ^ data[4]
            if checksum != 0:
                print("Checksum failed! Ignoring corrupt data.")
                continue

            quality = data[0] >> 2
            angle = ((data[1] | ((data[2] & 0x7F) << 8)) / 64.0)
            distance_raw = (data[3] | (data[4] << 8))

            # Test different distance scales
            distance_mm = distance_raw / 4.0
            distance_cm = distance_raw / 10.0
            distance_m = distance_raw / 1000.0

            print(f"Raw Distance: {distance_raw} | mm: {distance_mm} | cm: {distance_cm} | m: {distance_m}")

            # Use the correct distance scale
            distance = distance_m  # Adjust if needed

            if quality > 0 and 0 < distance < 6.0:  # Ensure valid distance
                x = distance * math.cos(math.radians(angle))
                y = distance * math.sin(math.radians(angle))
                points.append((x, y))

                # Apply median filter to remove noise
                if len(points) > 5:
                    distances = [p[0] for p in points]
                    median_distance = sorted(distances)[len(distances) // 2]
                    if abs(distance - median_distance) > 0.5:
                        print(f"Outlier detected: {distance}m (Median: {median_distance}m)")
                        continue

                self.screen.fill(BACKGROUND_COLOR)
                self.draw_grid()

                if not self.reference_detected and len(points) > 100:
                    self.reference_corner = self.detect_bottom_left_corner(points)
                    if self.reference_corner:
                        self.reference_detected = True
                        print(f"Reference corner detected at: {self.reference_corner}")

                if self.reference_corner:
                    pygame.draw.circle(self.screen, (0, 255, 0), self.world_to_screen(*self.reference_corner), 8)

                for px, py in points:
                    pygame.draw.circle(self.screen, POINT_COLOR, self.world_to_screen(px, py), 2)

                self.draw_lidar_position()
                pygame.display.flip()

                if len(points) > 1000:
                    points = points[-1000:]

    def calculate_lidar_position(self, points):
        if len(points) < 100 or not self.reference_corner:
            return None

        points_array = np.array(points[-100:])
        distances = np.sqrt(points_array[:, 0]**2 + points_array[:, 1]**2)
        avg_distance = np.mean(distances)
        return (0, 0)  # Keeping LIDAR at origin for now

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
