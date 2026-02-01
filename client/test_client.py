import cv2
import requests
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to file)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/predict", help="Inference Endpoint URL")
    args = parser.parse_args()

    URL = args.url

    # Handle numeric source (webcam)
    source = int(args.source) if args.source.isdigit() else args.source
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print(f"Streaming to {URL}...")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        # Encode frame to JPEG
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        start_time = time.time()
        try:
            # Send to Ray Serve
            response = requests.post(
                URL, 
                files={"file": ("frame.jpg", img_bytes, "image/jpeg")}
            )
            response.raise_for_status()
            result = response.json()
            
            # FPS Calculation (Round trip)
            latency = time.time() - start_time
            fps = 1.0 / latency if latency > 0 else 0
            
            # Visualize
            # Visualize Fusion
            detections = result.get("detections", [])
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                conf = det["conf"]
                label = det["label"]
                depth_score = det.get("depth_score", 0.0)

                # Color: Green
                color = (0, 255, 0)
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label + Depth
                text = f"{label} {conf:.2f} | D-Score: {depth_score:.1f}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Decode Depth Map
            depth_b64 = result.get("depth_map")
            if depth_b64:
                import base64
                depth_bytes = base64.b64decode(depth_b64)
                depth_np = np.frombuffer(depth_bytes, np.uint8)
                depth_img = cv2.imdecode(depth_np, cv2.IMREAD_GRAYSCALE)
                
                # Apply Colormap (Inferno looks cool for depth)
                depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
                cv2.imshow("Depth Anything Model", depth_color)

            # Show Windows
            cv2.imshow("Depth fused with Yolo", frame) # Reusing frame which now has boxes (Technically Fusion, but shows YOLO work)
            # Ideally we would show a clean frame for YOLO, but overlay is fine for demo


        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Request failed: {e}")
            # Keep trying even if one frame fails
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
