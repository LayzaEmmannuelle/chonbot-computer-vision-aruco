"""Project core: robots via ArUco, ball via YOLO, and navigation command to the ball.

Angle convention (same as image coordinates, with y pointing down):
  0 degrees   = robot front pointing RIGHT in the image
  +90 degrees = pointing DOWN
  -90 degrees = pointing UP
That is, a positive angle corresponds to a clockwise rotation seen from above.
"""
import math

import cv2

import config


def create_detector():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(config.DICTIONARY)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # more accurate corners
    return aruco.ArucoDetector(dictionary, params)


def normalize(angle):
    """Wraps any angle into the [-180, 180) range."""
    return (angle + 180.0) % 360.0 - 180.0


def detect_robots(detector, frame):
    """Returns a list of dicts: id, name, x, y, angle, corners."""
    corners, ids, _ = detector.detectMarkers(frame)
    robots = []
    if ids is None:
        return robots
    for c, id_ in zip(corners, ids.flatten()):
        id_ = int(id_)
        pts = c[0]                          # 4 corners starting from the marker's top-left corner
        cx, cy = pts.mean(axis=0)           # center = robot position
        dx, dy = pts[1] - pts[0]            # top edge of the marker = robot front
        angle = normalize(math.degrees(math.atan2(dy, dx)) + config.FRONT_OFFSET_DEGREES)
        robots.append({
            "id": id_,
            "name": config.ROBOTS.get(id_, f"unknown_{id_}"),
            "x": float(cx),
            "y": float(cy),
            "angle": float(angle),
            "corners": pts.tolist(),
        })
    return robots


def command(rx, ry, robot_angle, tx, ty, tol=None, stop_dist=None):
    """Decides the movement command for the robot to reach target (tx, ty).

    Returns (command, error_in_degrees, distance_in_pixels).
    command is "FORWARD", "RIGHT", "LEFT", or "STOP".
    """
    tol = config.TOLERANCE_DEGREES if tol is None else tol
    stop_dist = config.STOP_DISTANCE_PX if stop_dist is None else stop_dist

    dist = math.hypot(tx - rx, ty - ry)
    if dist < stop_dist:
        return "STOP", 0.0, dist

    target_angle = math.degrees(math.atan2(ty - ry, tx - rx))
    error = normalize(target_angle - robot_angle)
    if abs(error) <= tol:
        return "FORWARD", error, dist
    return ("RIGHT" if error > 0 else "LEFT"), error, dist


class Perception:
    """Combines ArUco (robots) and YOLO (ball only) in a single per-frame step."""

    def __init__(self, weights=None, yolo_every=2, conf=0.4, imgsz=640, max_lost_frames=15):
        self.detector = create_detector()
        self.ball_detector = None
        if weights:
            from ball_detector import BallDetector    # loads YOLO only if --weights is passed
            self.ball_detector = BallDetector(weights, conf=conf, imgsz=imgsz)
        self.yolo_every = max(1, yolo_every)        # runs YOLO every N frames (saves CPU/GPU)
        self.max_lost_frames = max_lost_frames      # frames without seeing the ball before forgetting it
        self.frame_count = 0
        self.ball = None
        self.frames_unseen = 0

    def process(self, frame):
        """Returns (robots, ball). Each robot gets a "to_ball" dict when the ball is known."""
        self.frame_count += 1
        robots = detect_robots(self.detector, frame)

        if self.ball_detector is not None and self.frame_count % self.yolo_every == 0:
            found = self.ball_detector.detect(frame)
            if found is not None:
                self.ball = found
                self.frames_unseen = 0
            else:
                self.frames_unseen += self.yolo_every
                if self.frames_unseen > self.max_lost_frames:
                    self.ball = None

        if self.ball is not None:
            for r in robots:
                cmd, error, dist = command(r["x"], r["y"], r["angle"], self.ball["x"], self.ball["y"])
                r["to_ball"] = {"command": cmd, "error": float(error), "dist": float(dist)}
        return robots, self.ball