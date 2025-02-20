import numpy as np
import pygame
import math
from rplidar import RPLidar
import time

# ----- SLAM and Map Parameters -----
MAP_SIZE = 8.0          # 8x8 meter area
SCALE = 100             # 100 pixels per meter for visualization
NUM_PARTICLES = 100     # Number of particles for the filter

# Initialize particles: each particle is (x, y, theta)
particles = np.empty((NUM_PARTICLES, 3))
particles[:, 0] = np.random.uniform(0, MAP_SIZE, NUM_PARTICLES)  # x positions
particles[:, 1] = np.random.uniform(0, MAP_SIZE, NUM_PARTICLES)  # y positions
particles[:, 2] = np.random.uniform(-math.pi, math.pi, NUM_PARTICLES)  # orientations

# Initialize uniform weights
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

# ----- LIDAR Setup -----
PORT_NAME = '/dev/ttyUSB0'  # Update to your system's port (e.g., 'COM3' on Windows)
BAUDRATE = 460800           # Lidar's baudrate
lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)

# ----- Pygame Setup -----
WINDOW_SIZE = (int(MAP_SIZE * SCALE), int(MAP_SIZE * SCALE))
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Particle Filter SLAM - Lidar Localization")

# ----- Utility Functions -----
def polar_to_cartesian(distance, angle):
    """Convert polar coordinates to Cartesian coordinates."""
    radians = math.radians(angle)
    x = distance * math.cos(radians)
    y = distance * math.sin(radians)
    return x, y

def motion_update(particles, delta_pose):
    """
    Update particle poses based on a motion model.
    delta_pose is (dx, dy, dtheta) estimated from scan matching.
    Here we add a small amount of noise to simulate uncertainty.
    """
    noise_std = np.array([0.005, 0.005, 0.002])
    for i in range(len(particles)):
        # Add noise to the motion update
        dx = delta_pose[0] + np.random.normal(0, noise_std[0])
        dy = delta_pose[1] + np.random.normal(0, noise_std[1])
        dtheta = delta_pose[2] + np.random.normal(0, noise_std[2])
        theta = particles[i, 2]
        # Apply the motion update in the particle's frame
        particles[i, 0] += dx * math.cos(theta) - dy * math.sin(theta)
        particles[i, 1] += dx * math.sin(theta) + dy * math.cos(theta)
        particles[i, 2] += dtheta
    return particles

def sensor_model(particle, scan_points):
    """
    Compute a likelihood for a given particle given the current scan.
    In a real system, you would compare the expected distances (from your map)
    with the measured distances. Here we use a dummy model that returns 1.0.
    """
    return 1.0

def update_weights(particles, weights, scan_points):
    """Update particle weights using the sensor model."""
    for i, particle in enumerate(particles):
        weights[i] = sensor_model(particle, scan_points)
    # Normalize weights
    weights += 1e-300
    weights /= np.sum(weights)
    return weights

def resample_particles(particles, weights):
    """Resample particles based on their weights."""
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[indices]

# ----- Main SLAM Loop -----
prev_scan = None
estimated_pose = np.array([MAP_SIZE / 2, MAP_SIZE / 2, 0])  # Start in the center

try:
    for scan in lidar.iter_scans():
        # Process the current scan: convert measurements to (x,y) points (in meters)
        scan_points = []
        for quality, angle, distance in scan:
            if 0 < distance < 5000:  # Accept distances within a valid range
                d = distance / 1000.0  # Convert from mm to meters
                x, y = polar_to_cartesian(d, angle)
                scan_points.append((x, y))
        scan_points = np.array(scan_points)

        # Estimate relative motion between scans using scan matching (placeholder)
        # In a real implementation, replace the following with an ICP or similar algorithm.
        if prev_scan is not None:
            # Dummy relative motion (e.g., small forward movement)
            delta_pose = np.array([0.01, 0.0, 0.001])
        else:
            delta_pose = np.array([0.0, 0.0, 0.0])
        prev_scan = scan_points

        # Particle filter motion update
        particles = motion_update(particles, delta_pose)

        # Particle filter sensor update
        weights = update_weights(particles, weights, scan_points)

        # Resample particles based on updated weights
        particles = resample_particles(particles, weights)

        # Estimate current pose as weighted average
        estimated_pose = np.average(particles, axis=0, weights=weights)

        # ----- Visualization -----
        screen.fill((0, 0, 0))
        # Draw particles (in yellow)
        for p in particles:
            px = int(p[0] * SCALE)
            py = int(WINDOW_SIZE[1] - p[1] * SCALE)
            pygame.draw.circle(screen, (255, 255, 0), (px, py), 2)
        # Draw estimated pose (in blue)
        ex = int(estimated_pose[0] * SCALE)
        ey = int(WINDOW_SIZE[1] - estimated_pose[1] * SCALE)
        pygame.draw.circle(screen, (0, 0, 255), (ex, ey), 5)
        pygame.display.flip()
        time.sleep(0.1)

        # Check for Pygame quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("SLAM process stopped.")
finally:
    lidar.stop()
    lidar.disconnect()
    pygame.quit()
