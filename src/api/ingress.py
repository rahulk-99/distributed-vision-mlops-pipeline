from fastapi import FastAPI, UploadFile, File
from ray import serve
from ray.serve.handle import DeploymentHandle
import numpy as np
import io
import ray
from PIL import Image
import cv2
from src.models.yolo_service import YOLODeployment
from src.models.depth_service import DepthDeployment

app = FastAPI()

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1})
@serve.ingress(app)
class APIIngress:
    def __init__(self, yolo_handle: DeploymentHandle, depth_handle: DeploymentHandle):
        self.yolo = yolo_handle
        self.depth = depth_handle

    @app.post("/predict")
    async def predict(self, file: UploadFile = File(...)):
        # 1. Decode image once (CPU operation)
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # 2. Parallel Inference
        # Ray Shared Memory (Plasma) handles zero-copy data transfer
        ref_yolo = self.yolo.remote(image_np)
        ref_depth = self.depth.remote(image_np)

        import time
        start_time = time.time()
        
        # 3. Gather Results
        detections = await ref_yolo
        depth_map = await ref_depth
        
        end_time = time.time()
        print(f"[Ingress] Parallel execution wait time: {(end_time - start_time) * 1000:.2f} ms")

        # 4. Sensor Fusion Logic
        final_results = []
        h, w = depth_map.shape
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            # Clamp coordinates to image bounds
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            # Crop depth map to the bounding box
            depth_crop = depth_map[y1:y2, x1:x2]
            
            depth_value = 0.0
            if depth_crop.size > 0:
                # Use median to be robust against outliers/edges
                depth_value = float(np.median(depth_crop))

            # Annotate detection with median depth score
            # Note: Depth Anything output is relative (non-metric)
            det["depth_score"] = depth_value
            final_results.append(det)

        # 5. Prepare Visualization Artifacts (For Demo)
        # Normalize depth map to 0-255 for visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = np.zeros_like(depth_map)
        
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # Encode to Base64
        import base64
        _, buffer = cv2.imencode('.jpg', depth_uint8)
        depth_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "detections": final_results,
            "depth_map": depth_b64
        }

# Entrypoint for 'serve run'
# This binds the deployments together into a single application
yolo_bind = YOLODeployment.bind()
depth_bind = DepthDeployment.bind()
ingress = APIIngress.bind(yolo_bind, depth_bind)
