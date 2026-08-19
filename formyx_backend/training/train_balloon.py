import os
import shutil
from ultralytics import YOLO

def main():
    # Paths
    dataset_yaml = "balloon_dataset/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"[-] Error: Dataset configuration not found at {dataset_yaml}")
        print("[-] Please run 'prepare_dataset.py' first.")
        return

    # Load pretrained YOLO11n (will download automatically if not present)
    print("[*] Loading pretrained YOLO11n model...")
    model = YOLO("yolo11n.pt")

    # Start training on CPU (since deployment target is CPU-only, and 74 images is very small)
    epochs = 30
    print(f"[*] Starting YOLO11n training on custom balloon dataset for {epochs} epochs...")
    
    try:
        model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,
            device="cpu",
            workers=4,
            project="runs",
            name="yolo11n_balloon",
            exist_ok=True
        )
        print("[+] Training completed successfully.")
        
        # Copy the trained weights to the workspace root as yolo11n_balloon.pt
        best_weights_src = "runs/yolo11n_balloon/weights/best.pt"
        best_weights_dst = "yolo11n_balloon.pt"
        
        if os.path.exists(best_weights_src):
            shutil.copy(best_weights_src, best_weights_dst)
            print(f"[+] Successfully copied fine-tuned weights to: {best_weights_dst}")
            print(f"[+] You can now run your balloon tracker script realsense_balloon_inference.py!")
        else:
            print(f"[-] Warning: Trained weights not found at {best_weights_src}")
            
    except Exception as e:
        print(f"[-] Error during training: {e}")

if __name__ == "__main__":
    main()
