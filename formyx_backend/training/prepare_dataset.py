import os
import csv
import ast
import shutil
import random

def main():
    # Paths
    csv_path = "archive/balloon-data.csv"
    images_dir = "archive/images"
    output_dir = "balloon_dataset"

    print(f"[*] Reading annotations from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"[-] Error: Annotation file not found at {csv_path}")
        return

    # Load CSV using built-in csv module
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f_csv:
        reader = csv.DictReader(f_csv)
        for row in reader:
            records.append(row)
            
    print(f"[+] Loaded {len(records)} entries from CSV.")

    # Create directories for training and validation splits
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Perform train/validation split manually (80% train, 20% validation)
    random.seed(42)
    random.shuffle(records)
    
    split_idx = int(len(records) * 0.8)
    train_data = records[:split_idx]
    val_data = records[split_idx:]
    print(f"[*] Splitting: {len(train_data)} training samples, {len(val_data)} validation samples.")

    def process_split(split_df, split_name):
        count = 0
        for row in split_df:
            fname = row["fname"]
            h_img = float(row["height"])
            w_img = float(row["width"])
            
            # Parse bounding boxes list from string literal representation
            try:
                bboxes = ast.literal_eval(row["bbox"])
            except Exception as e:
                print(f"[-] Error parsing bbox list for {fname}: {e}")
                continue
                
            # Copy image to respective target folder
            src_img = os.path.join(images_dir, fname)
            if not os.path.exists(src_img):
                print(f"[-] Warning: Image file not found: {src_img}")
                continue
                
            dst_img = os.path.join(output_dir, split_name, "images", fname)
            shutil.copy(src_img, dst_img)
            
            # Write labels in YOLO format (class x_center y_center width height)
            label_fname = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(output_dir, split_name, "labels", label_fname)
            
            with open(label_path, "w") as label_file:
                for box in bboxes:
                    xmin = float(box["xmin"])
                    ymin = float(box["ymin"])
                    xmax = float(box["xmax"])
                    ymax = float(box["ymax"])
                    
                    # Compute box width and height
                    box_w = xmax - xmin
                    box_h = ymax - ymin
                    
                    # Compute center coordinates
                    cx = xmin + box_w / 2.0
                    cy = ymin + box_h / 2.0
                    
                    # Normalize relative to image size
                    cx_norm = cx / w_img
                    cy_norm = cy / h_img
                    w_norm = box_w / w_img
                    h_norm = box_h / h_img
                    
                    # Write class_id 0 (balloon) and coordinates
                    label_file.write(f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            count += 1
        print(f"[+] Processed {count} images in {split_name} split.")

    print("\n[*] Copying training images and formatting labels...")
    process_split(train_data, "train")
    
    print("\n[*] Copying validation images and formatting labels...")
    process_split(val_data, "val")

    # Generate YOLO configuration yaml file (using absolute path)
    abs_output_dir = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""path: {abs_output_dir}
train: train/images
val: val/images

nc: 1
names:
  0: balloon
"""
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f_yaml:
        f_yaml.write(yaml_content)
        
    print(f"\n[+] Dataset preparation complete!")
    print(f"[+] YOLO config file created at: {yaml_path}")

if __name__ == "__main__":
    main()
