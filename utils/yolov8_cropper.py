from ultralytics import YOLO
from PIL import Image

class YoloCropper:
    def __init__(self, model_name='yolov8n.pt', conf_thresh=0.5):
        # Downloaded automatically if not present
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh

    def crop(self, img: Image.Image):
        """
        Runs YOLOv8 on the PIL image `img` and returns
        a list of (crop_image: PIL.Image, confidence: float).
        """
        # run inference (returns a list; we take the first)
        results = self.model.predict(img, imgsz=640, verbose=False)[0]
        crops = []
        # each `box` is [x1, y1, x2, y2]
        for box, conf in zip(results.boxes.xyxy, results.boxes.conf):
            if conf < self.conf_thresh:
                continue
            x1, y1, x2, y2 = map(int, box)
            crop = img.crop((x1, y1, x2, y2))
            crops.append((crop, float(conf)))
        return crops
