import RPi.GPIO as GPIO
import time

# L298N pins configuration for all wheels
# Front wheels
FRONT_IN1 = 5   # Front left wheel
FRONT_IN2 = 6   # Front left wheel
FRONT_IN3 = 13  # Front right wheel 
FRONT_IN4 = 19  # Front right wheel

# Rear wheels
REAR_IN1 = 17   # Rear left wheel
REAR_IN2 = 22   # Rear left wheel
REAR_IN3 = 23   # Rear right wheel
REAR_IN4 = 24   # Rear right wheel

# GPIO setup
GPIO.setmode(GPIO.BCM)  # Use BCM numbering scheme
GPIO.setwarnings(False)  # Disable warnings

# Set up motor pins
GPIO.setup(FRONT_IN1, GPIO.OUT)
GPIO.setup(FRONT_IN2, GPIO.OUT)
GPIO.setup(FRONT_IN3, GPIO.OUT)
GPIO.setup(FRONT_IN4, GPIO.OUT)
GPIO.setup(REAR_IN1, GPIO.OUT)
GPIO.setup(REAR_IN2, GPIO.OUT)
GPIO.setup(REAR_IN3, GPIO.OUT)
GPIO.setup(REAR_IN4, GPIO.OUT)

def forward():
    """Move all wheels forward"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.LOW)
    GPIO.output(FRONT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_IN3, GPIO.HIGH)
    GPIO.output(FRONT_IN4, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.HIGH)
    GPIO.output(REAR_IN2, GPIO.LOW)
    GPIO.output(REAR_IN3, GPIO.HIGH)
    GPIO.output(REAR_IN4, GPIO.LOW)
    
    print("Moving forward")

def backward():
    """Move all wheels backward"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_IN2, GPIO.LOW)
    GPIO.output(FRONT_IN3, GPIO.LOW)
    GPIO.output(FRONT_IN4, GPIO.HIGH)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.LOW)
    GPIO.output(REAR_IN2, GPIO.HIGH)
    GPIO.output(REAR_IN3, GPIO.LOW)
    GPIO.output(REAR_IN4, GPIO.HIGH)
    
    print("Moving backward")

def stop():
    """Stop all wheels"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.LOW)
    GPIO.output(FRONT_IN2, GPIO.LOW)
    GPIO.output(FRONT_IN3, GPIO.LOW)
    GPIO.output(FRONT_IN4, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.LOW)
    GPIO.output(REAR_IN2, GPIO.LOW)
    GPIO.output(REAR_IN3, GPIO.LOW)
    GPIO.output(REAR_IN4, GPIO.LOW)
    
    print("Stopped")

def strafe_left():
    """Move all wheels to strafe left (mecanum wheels feature)"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_IN2, GPIO.LOW)
    GPIO.output(FRONT_IN3, GPIO.HIGH)
    GPIO.output(FRONT_IN4, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.LOW)
    GPIO.output(REAR_IN2, GPIO.HIGH)
    GPIO.output(REAR_IN3, GPIO.LOW)
    GPIO.output(REAR_IN4, GPIO.HIGH)
    
    print("Strafing left")

def strafe_right():
    """Move all wheels to strafe right (mecanum wheels feature)"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.LOW)
    GPIO.output(FRONT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_IN3, GPIO.LOW)
    GPIO.output(FRONT_IN4, GPIO.HIGH)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.HIGH)
    GPIO.output(REAR_IN2, GPIO.LOW)
    GPIO.output(REAR_IN3, GPIO.HIGH)
    GPIO.output(REAR_IN4, GPIO.LOW)
    
    print("Strafing right")

def rotate_clockwise():
    """Rotate the robot clockwise"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.LOW)
    GPIO.output(FRONT_IN2, GPIO.HIGH)
    GPIO.output(FRONT_IN3, GPIO.LOW)
    GPIO.output(FRONT_IN4, GPIO.HIGH)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.LOW)
    GPIO.output(REAR_IN2, GPIO.HIGH)
    GPIO.output(REAR_IN3, GPIO.LOW)
    GPIO.output(REAR_IN4, GPIO.HIGH)
    
    print("Rotating clockwise")

def rotate_counterclockwise():
    """Rotate the robot counter-clockwise"""
    # Front wheels
    GPIO.output(FRONT_IN1, GPIO.HIGH)
    GPIO.output(FRONT_IN2, GPIO.LOW)
    GPIO.output(FRONT_IN3, GPIO.HIGH)
    GPIO.output(FRONT_IN4, GPIO.LOW)
    
    # Rear wheels
    GPIO.output(REAR_IN1, GPIO.HIGH)
    GPIO.output(REAR_IN2, GPIO.LOW)
    GPIO.output(REAR_IN3, GPIO.HIGH)
    GPIO.output(REAR_IN4, GPIO.LOW)
    
    print("Rotating counter-clockwise")

def test_sequence():
    """Run a test sequence to verify all movements"""
    print("Starting mecanum wheels test sequence...")
    
    # Forward test
    print("Testing forward movement...")
    forward()
    time.sleep(3)
    
    # Stop
    stop()
    time.sleep(1)
    
    # Backward test
    print("Testing backward movement...")
    backward()
    time.sleep(3)
    
    # Stop
    stop()
    time.sleep(1)
    
    # Strafe left test
    print("Testing strafe left movement...")
    strafe_left()
    time.sleep(3)
    
    # Stop
    stop()
    time.sleep(1)
    
    # Strafe right test
    print("Testing strafe right movement...")
    strafe_right()
    time.sleep(3)
    
    # Stop
    stop()
    time.sleep(1)
    
    # Rotate clockwise test
    print("Testing clockwise rotation...")
    rotate_clockwise()
    time.sleep(3)
    
    # Stop
    stop()
    time.sleep(1)
    
    # Rotate counter-clockwise test
    print("Testing counter-clockwise rotation...")
    rotate_counterclockwise()
    time.sleep(3)
    
    # Stop again
    stop()
    
    print("Test sequence completed")

try:
    test_sequence()
    
except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    # Clean up GPIO
    GPIO.cleanup()
    print("GPIO cleaned up")
