"""ArUco project configuration.
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160 for 4K
Edit ONLY this file to change camera settings, names, colors, sizes, and tolerances.
"""
import cv2

# USB camera configuration (switch CAMERA_INDEX between 0 and 1 if needed)
CAMERA_INDEX = 1
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_CODEC = "MJPG"

# ArUco dictionary for YOUR markers
DICTIONARY_NAME = "DICT_4X4_50"
DICTIONARY = getattr(cv2.aruco, DICTIONARY_NAME)

# Marker ID (0 to 6) -> robot name. Make sure each ID matches the
# marker attached to each robot!
ROBOTS = {
    0: "Decocco",
    1: "Gorlami",
    2: "Kombiceptor",
    3: "Kombusprime",
    4: "Margheriti",
    5: "Samu",
    6: "Tristeza",
}

# If the marker is attached rotated relative to the robot's front, adjust here
# (in degrees). Example: marker rotated 90 degrees to the right -> -90.
FRONT_OFFSET_DEGREES = -90.0

# Control parameters (used in perception.command)
TOLERANCE_DEGREES = 15      # below this error, the robot moves forward
STOP_DISTANCE_PX = 80       # target reached threshold (in pixels, depends on camera resolution)