import numpy as np
import torch
from ray import serve
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0.5})
class DepthDeployment:
    def __init__(self):
        print("Loading Depth Anything model...")
        model_id = "LiheYoung/depth-anything-small-hf"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        print(f"Depth Anything model loaded on {self.device}.")

    def __call__(self, image: np.ndarray):
        """
        Inference handler.
        Args:
            image (np.ndarray): RGB image [H, W, 3]
        Returns:
            np.ndarray: Depth map matched to original resolution.
        """
        # Transformers pipeline expects PIL or Tensor
        pil_image = Image.fromarray(image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        import time
        
        # Ensure previous ops are done
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # Synchronize again to wait for completion
            if self.device == "cuda":
                torch.cuda.synchronize()
                
        end_time = time.time()
        
        print(f"[Depth] Inference time (Synced): {(end_time - start_time) * 1000:.2f} ms")
        print(f"[Depth] Output Stats - Shape: {predicted_depth.shape}, Mean: {predicted_depth.mean().item():.4f}")

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=pil_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        # Return simple numpy array
        return prediction.squeeze().cpu().numpy()
