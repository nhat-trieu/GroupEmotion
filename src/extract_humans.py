"""
Extract human patches using YOLOv8 person detection
Saves features as .npz files (humans: N×D, boxes: N×4)
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try importing ultralytics YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not available. Install with: pip install ultralytics")

# Import feature extractor
import torchvision.models as models
from torchvision import transforms


def setup_logging(out_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(out_dir, 'process_humans.log')
    error_file = os.path.join(out_dir, 'errors_humans.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return error_file


def load_feature_extractor(device):
    """Load ResNet50 for feature extraction"""
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=True)
    # Remove final FC layer to get 2048-d features
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform


def detect_humans_yolo(img_rgb, model, max_persons=16):
    """
    Detect humans using YOLOv8
    Returns: list of boxes [x1, y1, x2, y2] and confidences
    """
    results = model(img_rgb, classes=[0], verbose=False)  # class 0 = person
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], []
    
    boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4]
    confs = results[0].boxes.conf.cpu().numpy()   # [N]
    
    # Sort by confidence
    idx = np.argsort(confs)[::-1][:max_persons]
    boxes = boxes[idx]
    confs = confs[idx]
    
    return boxes, confs


def extract_features(crops, feature_model, transform, device):
    """Extract features from cropped human patches"""
    if len(crops) == 0:
        return np.zeros((0, 2048), dtype=np.float32)
    
    features = []
    for crop in crops:
        crop_pil = Image.fromarray(crop)
        crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = feature_model(crop_tensor)
            feat = feat.squeeze().cpu().numpy()
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def process_image(img_path, detector, feature_model, transform, device, 
                  max_persons, error_file):
    """Process single image and extract human features"""
    try:
        # Read image
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            error_msg = f"Failed to read image: {img_path}"
            logging.error(error_msg)
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + '\n')
            return None, None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Detect humans
        boxes, confs = detect_humans_yolo(img_rgb, detector, max_persons)
        
        if len(boxes) == 0:
            # No humans detected
            return np.zeros((0, 2048), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        # Crop human patches
        crops = []
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img_rgb[y1:y2, x1:x2]
            crops.append(crop)
            valid_boxes.append([x1, y1, x2, y2])
        
        if len(crops) == 0:
            return np.zeros((0, 2048), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        # Extract features
        features = extract_features(crops, feature_model, transform, device)
        boxes_array = np.array(valid_boxes, dtype=np.float32)
        
        return features, boxes_array
        
    except Exception as e:
        error_msg = f"Error processing {img_path}: {str(e)}"
        logging.error(error_msg)
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Extract human patches and features')
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Path to image folder')
    parser.add_argument('--out_dir', type=str, required=True, 
                        help='Output directory for .npz files')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--max_persons', type=int, default=16, 
                        help='Maximum number of persons per image')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model size (yolov8n/s/m/l/x.pt)')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # Setup directories
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    
    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    error_file = setup_logging(out_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() 
                          and args.device == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load models
    logging.info("Loading YOLOv8 detector...")
    detector = YOLO(args.yolo_model)
    
    logging.info("Loading ResNet50 feature extractor...")
    feature_model, transform = load_feature_extractor(device)
    
    logging.info("Models loaded successfully")
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process images
    stats = {
        'total': len(image_files),
        'with_humans': 0,
        'no_humans': 0,
        'errors': 0
    }
    
    for img_path in tqdm(image_files, desc="Extracting humans"):
        features, boxes = process_image(
            img_path, detector, feature_model, transform, 
            device, args.max_persons, error_file
        )
        
        if features is None:
            stats['errors'] += 1
            continue
        
        # Save results
        out_file = out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_file, humans=features, boxes=boxes)
        
        if len(features) > 0:
            stats['with_humans'] += 1
        else:
            stats['no_humans'] += 1
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*50)
    logging.info(f"Total images: {stats['total']}")
    logging.info(f"Images with humans: {stats['with_humans']}")
    logging.info(f"Images without humans: {stats['no_humans']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info(f"Output saved to: {out_dir}")
    logging.info("="*50)


if __name__ == '__main__':
    main()