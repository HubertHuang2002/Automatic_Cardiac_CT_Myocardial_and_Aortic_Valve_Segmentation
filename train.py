try:
    import ultralytics
    print("Ultralytics is already installed")
except ImportError:
    print("installing YOLOv8 (ultralytics)")
    !pip install ultralytics
    print("installation complete")

import os
import glob
import shutil
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm.notebook import tqdm 
from ultralytics import YOLO


# Default working directory (for Colab)
BASE_DIR = os.getcwd()
DATA_ROOT = os.path.join(BASE_DIR, '/content/data')

"""
suggested file structure:
/content/data/training_image/
/content/data/training_label/
/content/data/testing_image/
"""
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'training_image')
TRAIN_LBL_DIR = os.path.join(DATA_ROOT, 'training_label')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'testing_image')

YOLO_DATASET_DIR = os.path.join(BASE_DIR, 'yolo_dataset')
OUTPUT_FILE = 'merged.txt'

# Set random seed for reproducibility
SEED = 42
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# Dataset preparation 
def prepare_dataset(val_split=0.2):
    # Clean raw dataset and convert to YOLO format
    print(f"Checking dataset source: {TRAIN_IMG_DIR}")
    
    if not os.path.exists(TRAIN_IMG_DIR):
        print("Error: 'training_image' folder not found")
        return False

    # Create YOLO directory structure
    dirs = ['train', 'val']
    for d in dirs:
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'images', d), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DATASET_DIR, 'labels', d), exist_ok=True)

    # Recursively search for PNG images inside patientXXXX folders
    all_image_paths = glob.glob(os.path.join(TRAIN_IMG_DIR, '**', '*.png'), recursive=True)
    
    print(f"Total images found: {len(all_image_paths)}")
    if len(all_image_paths) == 0:
        print("Warning: No .png files found")
        return False
    
    random.shuffle(all_image_paths)
    
    split_idx = int(len(all_image_paths) * (1 - val_split))
    train_imgs = all_image_paths[:split_idx]
    val_imgs = all_image_paths[split_idx:]
    
    def process_files(img_list, split_name):
        for img_path in tqdm(img_list, desc=f"處理 {split_name} 資料集"):
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            
            # Identify patient folder (patientXXXX)
            patient_id = os.path.basename(os.path.dirname(img_path))
            
            # Label path
            label_path = os.path.join(TRAIN_LBL_DIR, patient_id, f"{stem}.txt")
            
            # Copy image
            dst_img_path = os.path.join(YOLO_DATASET_DIR, 'images', split_name, filename)
            shutil.copy2(img_path, dst_img_path)
            
            # Copy label if exists
            dst_lbl_path = os.path.join(YOLO_DATASET_DIR, 'labels', split_name, f"{stem}.txt")
            if os.path.exists(label_path):
                shutil.copy2(label_path, dst_lbl_path)
                
    process_files(train_imgs, 'train')
    process_files(val_imgs, 'val')
    return True


# Create YOLO Data YAML file
def create_data_yaml():
    yaml_content = {
        'path': YOLO_DATASET_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'Aortic Valve'}
    }
    
    yaml_path = os.path.join(BASE_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path

# Train YOLO Model
def train_model(yaml_path):
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: GPU not detected")

    # Load YOLOv8 Medium model
    model = YOLO('yolov8m.pt') 
    
    print("\nStarting training")
    results = model.train(
        data=yaml_path,
        epochs=50,              
        imgsz=512,
        batch=16,
        patience=10,           
        project='Aortic_Project',
        name='train_run',
        exist_ok=True,
        verbose=True,
        fliplr=0.5,             
        flipud=0.0,           
        mosaic=1.0,            
    )
    return model


# Inference and generate merged.txt
def predict_and_generate(model):
    print("\nRunning inference on test set...")
    
    test_image_paths = glob.glob(os.path.join(TEST_IMG_DIR, '**', '*.png'), recursive=True)
    print(f"Total test images: {len(test_image_paths)}")
    
    predictions_list = []
    
    # Use low confidence threshold to retain all boxes
    conf_threshold = 0.001 
    
    for img_path in tqdm(test_image_paths, desc="Predicting"):
        results = model.predict(
            source=img_path,
            imgsz=512,
            conf=conf_threshold,
            iou=0.5,
            verbose=False
        )
        
        result = results[0]
        filename = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(filename)[0]
        
        if len(result.boxes) == 0:
            continue
            
        for box in result.boxes:
            # Convert into integer pixel coordinates
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            x1, y1, x2, y2 = map(lambda x: int(round(x)), xyxy)
            
            # boundary constraint
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(512, x2), min(512, y2)
            
            # Format: filename class score x1 y1 x2 y2
            line = f"{img_name_no_ext} {cls} {conf:.4f} {x1} {y1} {x2} {y2}"
            predictions_list.append(line)
      
    # Save output file
    with open(OUTPUT_FILE, 'w') as f:
        for line in predictions_list:
            f.write(line + '\n')
            
    print(f"\nOutput saved at: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    # Check dataset existence
    if os.path.exists(TRAIN_IMG_DIR):
        # Prepare dataset
        if not os.path.exists(os.path.join(YOLO_DATASET_DIR, 'images', 'train')):
            success = prepare_dataset(val_split=0.2)
            if not success:
                print("Dataset preparation failed")
                exit()
        
        # Generate YAML
        yaml_path = create_data_yaml()
        
        # Train model
        model = train_model(yaml_path)
        
        # Run inference
        predict_and_generate(model)
    else:
        print("Error: Dataset path not found")