import numpy as np
from ray import serve
from ultralytics import YOLO

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5})
class YOLODeployment:
    def __init__(self):
        # Initialize YOLOv8 (small variant)
        print("Loading YOLOv8n model...")
        self.model = YOLO("yolov8n.pt") 
        print(f"YOLOv8n model loaded. Device: {self.model.device}")

    def __call__(self, image: np.ndarray):
        """
        Inference handler.
        Args:
            image (np.ndarray): RGB image [H, W, 3]
        Returns:
            list: Detections with bbox, confidence, and class ID.
        """
        import time
        start_time = time.time()
        results = self.model(image)
        end_time = time.time()
        print(f"[YOLO] Inference time: {(end_time - start_time) * 1000:.2f} ms")
        
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                # Filter for 'person' class only
                label = result.names[int(box.cls[0])]
                if label == "person":
                    detections.append({
                        "bbox": box.xyxy[0].tolist(), # [x1, y1, x2, y2]
                        "conf": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                        "label": label
                    })
        
        return detections
