"""Flask server with the /perception/status endpoint (robots via ArUco, ball via YOLO).

  python server.py --weights best.pt
  python server.py --video test_aruco.mp4 --loop     # test without camera

Open in browser:  http://localhost:5000/perception/status
"""
import argparse
import threading
import time

import cv2
from flask import Flask, jsonify

import config
try:
    import perception
except ImportError:
    import percepcao as perception

app = Flask(__name__)
_lock = threading.Lock()
_state = {"timestamp": None, "frame_id": 0, "fps": 0.0, "ball": None, "robots": []}

ROBOT_FIELDS = ("id", "name", "x", "y", "angle", "to_ball")


def camera_loop(args):
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not args.video:
        codec = getattr(config, "CAMERA_CODEC", "MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    system = perception.Perception(args.weights, args.yolo_every, args.conf, args.imgsz)

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if args.video and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        robots, ball = system.process(frame)
        n += 1
        public_robots = [{k: r[k] for k in ROBOT_FIELDS if k in r} for r in robots]
        public_ball = None if ball is None else {
            "x": ball["x"], "y": ball["y"], "conf": ball["conf"]}
        with _lock:
            _state["timestamp"] = time.time()
            _state["frame_id"] = n
            _state["robots"] = public_robots
            _state["ball"] = public_ball
            if n % 10 == 0:
                _state["fps"] = 10 / (time.time() - t0)
                t0 = time.time()
        if args.video:
            time.sleep(1 / 30)      # simulates 30 fps on recorded video
    cap.release()


@app.get("/perception/status")
def status():
    with _lock:
        return jsonify(_state)


@app.get("/health")
def health():
    return jsonify({"ok": True})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=getattr(config, "CAMERA_INDEX", 1))
    ap.add_argument("--video")
    ap.add_argument("--loop", action="store_true", help="loop video playback (for testing)")
    ap.add_argument("--width", type=int, default=getattr(config, "CAMERA_WIDTH", 3840))
    ap.add_argument("--height", type=int, default=getattr(config, "CAMERA_HEIGHT", 2160))
    ap.add_argument("--weights", help="YOLO .pt weights file (ball only)")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-every", type=int, default=2, help="run YOLO every N frames")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    threading.Thread(target=camera_loop, args=(args,), daemon=True).start()
    app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()