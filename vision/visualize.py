"""Displays the camera (or a video) with robots (ArUco) and the ball (YOLO).

  python visualize.py                                # uses 4K camera / config.py
  python visualize.py --weights best.pt              # ArUco + ball via YOLO
  python visualize.py --video test_aruco.mp4

Without --weights, you can click on the image to simulate the ball.
Keys: q = quit | c = clear clicked target | s = save frame
"""
import argparse
import math
import time

import cv2

import config
try:
    import perception
except ImportError:
    import percepcao as perception

BOX_COLOR = (0, 200, 0)
ARROW_COLOR = (0, 140, 255)
TARGET_COLOR = (0, 0, 255)
BALL_COLOR = (0, 255, 255)
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "ArUco - Press [Q] to quit"


def open_source(args):
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not args.video:
        codec = getattr(config, "CAMERA_CODEC", "MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def draw_text(frame, msg, pos, color=(255, 255, 255), scale=1.0, thickness=2):
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw(frame, robots, ball, click_target, target_id):
    ui_scale = max(1.0, frame.shape[1] / 1280.0)
    thick = max(2, int(2 * ui_scale))

    for r in robots:
        pts = [(int(x), int(y)) for x, y in r["corners"]]
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i + 1) % 4], BOX_COLOR, thick)
        cx, cy = int(r["x"]), int(r["y"])
        side = max(abs(pts[1][0] - pts[0][0]), abs(pts[1][1] - pts[0][1]), 20)
        fx = int(cx + side * 1.2 * math.cos(math.radians(r["angle"])))
        fy = int(cy + side * 1.2 * math.sin(math.radians(r["angle"])))
        cv2.arrowedLine(frame, (cx, cy), (fx, fy), ARROW_COLOR, thick + 1, tipLength=0.3)
        ty = max(30, min(p[1] for p in pts) - int(14 * ui_scale))
        draw_text(frame, f'{r["name"]} (id {r["id"]}) {r["angle"]:.0f} deg',
                  (cx - int(60 * ui_scale), ty), scale=0.6 * ui_scale, thickness=thick)

    # target = ball (YOLO) if detected; otherwise the clicked point
    target = None
    if ball is not None:
        target = (int(ball["x"]), int(ball["y"]))
        radius = int(max(ball["width"], ball["height"]) / 2) + 4
        cv2.circle(frame, target, radius, BALL_COLOR, thick)
        draw_text(frame, f'ball {ball["conf"]:.2f}',
                  (target[0] - int(30 * ui_scale), target[1] - radius - 8),
                  BALL_COLOR, scale=0.6 * ui_scale, thickness=thick)
    elif click_target is not None:
        target = click_target
        cv2.drawMarker(frame, target, TARGET_COLOR, cv2.MARKER_CROSS, int(25 * ui_scale), thick)

    if target is not None:
        target_robots = sorted(
            [r for r in robots if target_id is None or r["id"] == target_id],
            key=lambda r: r["id"],
        )
        for idx, robot in enumerate(target_robots):
            cmd, error, dist = perception.command(robot["x"], robot["y"], robot["angle"], target[0], target[1])
            cv2.line(frame, (int(robot["x"]), int(robot["y"])), target, TARGET_COLOR, max(1, thick - 1))
            y_txt = int((75 + idx * 35) * ui_scale)
            draw_text(frame, f'{robot["name"]}: {cmd}  (error {error:.0f} deg, dist {dist:.0f}px)',
                      (int(15 * ui_scale), y_txt), (0, 255, 255),
                      scale=0.8 * ui_scale, thickness=thick)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=getattr(config, "CAMERA_INDEX", 1))
    ap.add_argument("--video")
    ap.add_argument("--width", type=int, default=getattr(config, "CAMERA_WIDTH", 3840))
    ap.add_argument("--height", type=int, default=getattr(config, "CAMERA_HEIGHT", 2160))
    ap.add_argument("--weights", help="YOLO .pt weights file (ball only)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-every", type=int, default=2, help="run YOLO every N frames")
    ap.add_argument("--robot", type=int, default=None, help="ID of the robot receiving target commands")
    ap.add_argument("--no-window", action="store_true", help="print to terminal only")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    args = ap.parse_args()

    cap = open_source(args)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video. Check CAMERA_INDEX in config.py.")
    system = perception.Perception(args.weights, args.yolo_every, args.conf, args.imgsz)

    state = {"target": None, "frame_shape": (args.height, args.width)}
    if not args.no_window:
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                h, w = state["frame_shape"]
                state["target"] = (int(x * w / WINDOW_WIDTH), int(y * h / WINDOW_HEIGHT))
        cv2.namedWindow(WINDOW_TITLE)
        cv2.setMouseCallback(WINDOW_TITLE, on_click)

    n, t0, fps = 0, time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        state["frame_shape"] = frame.shape[:2]
        robots, ball = system.process(frame)
        n += 1
        if n % 10 == 0:
            fps = 10 / (time.time() - t0)
            t0 = time.time()

        if args.no_window:
            if n % 10 == 0 or n == 1:
                summary = "; ".join(f'{r["name"]} x={r["x"]:.0f} y={r["y"]:.0f} ang={r["angle"]:.0f}' for r in robots)
                b = f'ball=({ball["x"]:.0f},{ball["y"]:.0f})' if ball else "ball=not seen"
                print(f"frame {n}: {len(robots)} robot(s) | {b} | {summary}")
        else:
            ui_scale = max(1.0, frame.shape[1] / 1280.0)
            draw(frame, robots, ball, state["target"], args.robot)
            draw_text(frame, f"{fps:.0f} FPS | {frame.shape[1]}x{frame.shape[0]} | {len(robots)} marker(s)",
                      (int(15 * ui_scale), int(35 * ui_scale)),
                      scale=0.8 * ui_scale, thickness=max(2, int(2 * ui_scale)))
            draw_text(frame, "Press [Q] to quit",
                      (int(15 * ui_scale), frame.shape[0] - int(20 * ui_scale)),
                      color=(200, 200, 200), scale=0.65 * ui_scale, thickness=max(2, int(2 * ui_scale)))
            cv2.imshow(WINDOW_TITLE, cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                state["target"] = None
            if key == ord("s"):
                filename = f"frame_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print("saved:", filename)
        if args.max_frames and n >= args.max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()