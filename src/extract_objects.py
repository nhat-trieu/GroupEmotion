"""
Extract salient objects using YOLOv8 object detection
Saves features as .npz files (objects: K×D, boxes: K×4)
Following Graph Neural Networks for Image Understanding paper approach
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

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

import torchvision.models as models
from torchvision import transforms


def setup_logging(out_dir):
    """Setup logging"""
    log_file = os.path.join(out_dir, 'process_objects.log')
    error_file = os.path.join(out_dir, 'errors_objects.txt')
    
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
    """Load SENet/ResNet50 for object feature extraction"""
    # Use ResNet50 as alternative to SENet-154
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform


def detect_objects_yolo(img_rgb, model, max_objects=16, conf_threshold=0.25):
    """
    Detect salient objects using YOLOv8
    Returns top-K objects by confidence
    """
    results = model(img_rgb, verbose=False, conf=conf_threshold)
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return [], [], []
    
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    
    # Sort by confidence and take top-K
    idx = np.argsort(confs)[::-1][:max_objects]
    
    return boxes[idx], confs[idx], classes[idx]


def extract_features(crops, feature_model, transform, device):
    """Extract features from object crops"""
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
                  max_objects, error_file):
    """Process single image and extract object features"""
    try:
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            error_msg = f"Failed to read image: {img_path}"
            logging.error(error_msg)
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + '\n')
            return None, None, None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        
        # Detect objects
        boxes, confs, classes = detect_objects_yolo(img_rgb, detector, max_objects)
        
        if len(boxes) == 0:
            return (np.zeros((0, 2048), dtype=np.float32), 
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))
        
        # Crop objects
        crops = []
        valid_boxes = []
        valid_classes = []
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = img_rgb[y1:y2, x1:x2]
            crops.append(crop)
            valid_boxes.append([x1, y1, x2, y2])
            valid_classes.append(cls)
        
        if len(crops) == 0:
            return (np.zeros((0, 2048), dtype=np.float32), 
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))
        
        # Extract features
        features = extract_features(crops, feature_model, transform, device)
        boxes_array = np.array(valid_boxes, dtype=np.float32)
        classes_array = np.array(valid_classes, dtype=np.int32)
        
        return features, boxes_array, classes_array
        
    except Exception as e:
        error_msg = f"Error processing {img_path}: {str(e)}"
        logging.error(error_msg)
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description='Extract object features')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'])
    parser.add_argument('--max_objects', type=int, default=16,
                        help='Max objects per image (paper uses 16)')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt')
    
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("ERROR: ultralytics not installed")
        sys.exit(1)
    
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    
    if not image_dir.exists():
        print(f"Error: {image_dir} not found")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    error_file = setup_logging(out_dir)
    
    device = torch.device(args.device if torch.cuda.is_available() 
                          and args.device == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load models
    logging.info("Loading YOLOv8...")
    detector = YOLO(args.yolo_model)
    
    logging.info("Loading feature extractor...")
    feature_model, transform = load_feature_extractor(device)
    
    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    logging.info(f"Found {len(image_files)} images")
    
    stats = {'total': len(image_files), 'with_objects': 0, 
             'no_objects': 0, 'errors': 0}
    
    for img_path in tqdm(image_files, desc="Extracting objects"):
        features, boxes, classes = process_image(
            img_path, detector, feature_model, transform,
            device, args.max_objects, error_file
        )
        
        if features is None:
            stats['errors'] += 1
            continue
        
        out_file = out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_file, objects=features, boxes=boxes, classes=classes)
        
        if len(features) > 0:
            stats['with_objects'] += 1
        else:
            stats['no_objects'] += 1
    
    # Summary
    logging.info("\n" + "="*50)
    logging.info("OBJECT EXTRACTION SUMMARY")
    logging.info("="*50)
    logging.info(f"Total: {stats['total']}")
    logging.info(f"With objects: {stats['with_objects']}")
    logging.info(f"No objects: {stats['no_objects']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info("="*50)


if __name__ == '__main__':
    main()