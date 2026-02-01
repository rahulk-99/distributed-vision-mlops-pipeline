import pytest
import numpy as np
import ray
from ray import serve
from src.models.yolo_service import YOLODeployment
from src.models.depth_service import DepthDeployment

def test_models():
    """
    Integration test checking if models run correctly within Ray Serve.
    """
    # 1. Setup local Ray instance
    print("Initializing Ray...")
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    serve.start(detached=False)

    mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        # 2. Test YOLO
        print("\n--- Testing YOLO Deployment ---")
        # serve.run takes a bound deployment and deploys it locally
        # It returns a DeploymentHandle
        yolo_handle = serve.run(YOLODeployment.bind(), name="yolo_test", route_prefix="/yolo")
        
        detections = yolo_handle.remote(mock_image).result()
        
        assert isinstance(detections, list)
        print(f"YOLO Success. Output type: {type(detections)}")

        # 3. Test Depth
        print("\n--- Testing Depth Deployment ---")
        depth_handle = serve.run(DepthDeployment.bind(), name="depth_test", route_prefix="/depth")
        
        # Note: The first run might be slow due to model loading (downloading if not cached)
        depth_map = depth_handle.remote(mock_image).result()
        
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.shape == (480, 640)
        print(f"Depth Success. Output shape: {depth_map.shape}")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()

if __name__ == "__main__":
    test_models()
