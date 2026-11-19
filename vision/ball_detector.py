"""Ball detector using YOLO (ball only; robots are tracked via ArUco)."""


class BallDetector:
    def __init__(self, weights, conf=0.4, imgsz=640, class_name="ball"):
        from ultralytics import YOLO          # imported here so ArUco works without YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

        # Finds the "ball" class ID by NAME (not by index),
        # so it works with both a multi-class best.pt and a ball-only model.
        names = self.model.names                        # {0: 'Decocco', ..., 6: 'ball', ...}
        ids = [i for i, n in names.items() if str(n).lower() == class_name.lower()]
        if ids:
            self.classes = ids
        else:
            self.classes = None                         # uses all classes in the model
            if len(names) > 1:
                print(f'WARNING: model does not contain the "{class_name}" class. '
                      f"Classes: {list(names.values())}. Using all classes.")

    def detect(self, frame):
        """Returns {x, y, conf, width, height} for the highest-confidence ball, or None."""
        res = self.model(frame, conf=self.conf, imgsz=self.imgsz,
                         classes=self.classes, verbose=False)[0]
        if len(res.boxes) == 0:
            return None
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        return {
            "x": (x1 + x2) / 2,
            "y": (y1 + y2) / 2,
            "conf": float(res.boxes.conf[i]),
            "width": x2 - x1,
            "height": y2 - y1,
        }