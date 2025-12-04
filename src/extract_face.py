"""
Face Feature Extraction Script
Extract 512-d embeddings from group images using MTCNN + InceptionResnetV1
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
from facenet_pytorch import MTCNN, InceptionResnetV1
import warnings
warnings.filterwarnings('ignore')


def setup_logging(out_dir):
    """Setup logging to file and console"""
    log_file = os.path.join(out_dir, 'process.log')
    error_file = os.path.join(out_dir, 'errors.txt')
    
    # Setup main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return error_file


def normalize_embedding(embedding):
    """L2 normalize embedding vector"""
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    return embedding / (norm + 1e-10)


def process_image(img_path, mtcnn, model, device, error_file):
    """
    Process single image: detect faces, extract embeddings
    Returns: faces_array (N,512), boxes_array (N,4)
    """
    try:
        # Read image with cv2
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            error_msg = f"Failed to read image: {img_path}"
            logging.error(error_msg)
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + '\n')
            return None, None
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Detect faces with MTCNN
        boxes, probs = mtcnn.detect(img_pil)
        
        # No faces detected
        if boxes is None:
            return np.zeros((0, 512), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        # Extract face crops and get embeddings
        faces_list = []
        valid_boxes = []
        
        for box in boxes:
            # Crop face region
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_rgb.shape[1], x2), min(img_rgb.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            face_crop = img_rgb[y1:y2, x1:x2]
            
            # Convert to PIL and resize to 160x160 for InceptionResnetV1
            face_pil = Image.fromarray(face_crop)
            face_pil = face_pil.resize((160, 160), Image.BILINEAR)
            
            # Convert to tensor and normalize [-1, 1]
            face_tensor = torch.tensor(np.array(face_pil)).permute(2, 0, 1).float()
            face_tensor = (face_tensor - 127.5) / 128.0
            face_tensor = face_tensor.unsqueeze(0).to(device)
            
            # Get embedding
            with torch.no_grad():
                embedding = model(face_tensor).cpu().numpy() # (1, 512)
            
            faces_list.append(embedding[0])
            valid_boxes.append(box)
        
        if len(faces_list) == 0:
            return np.zeros((0, 512), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        
        # Stack and normalize embeddings
        faces_array = np.stack(faces_list, axis=0)
        faces_array = normalize_embedding(faces_array)
        boxes_array = np.array(valid_boxes, dtype=np.float32)
        
        return faces_array, boxes_array
        
    except Exception as e:
        error_msg = f"Error processing {img_path}: {str(e)}"
        logging.error(error_msg)
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
        return None, None


def main():
    parser = argparse.ArgumentParser(description='Extract face embeddings from group images')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for .npz files')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--min_face_size', type=int, default=20, 
                        help='Minimum face size for detection')
    
    args = parser.parse_args()
    
    # Check and create directories
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    
    if not image_dir.exists():
        print(f"Error: Image directory does not exist: {image_dir}")
        sys.exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    error_file = setup_logging(out_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Initialize models
    logging.info("Loading MTCNN and InceptionResnetV1 models...")
    mtcnn = MTCNN(
        keep_all=True,
        device=device,
        min_face_size=args.min_face_size,
        thresholds=[0.6, 0.7, 0.7]
    )
    
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logging.info("Models loaded successfully")
    
    # Get list of image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process images
    stats = {
        'total': len(image_files),
        'with_faces': 0,
        'no_faces': 0,
        'errors': 0
    }
    
    for img_path in tqdm(image_files, desc="Processing images"):
        faces, boxes = process_image(img_path, mtcnn, model, device, error_file)
        
        if faces is None:
            stats['errors'] += 1
            continue
        
        # Save results
        out_file = out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_file, faces=faces, boxes=boxes)
        
        if len(faces) > 0:
            stats['with_faces'] += 1
        else:
            stats['no_faces'] += 1
    
    # Print summary
    logging.info("\n" + "="*50)
    logging.info("PROCESSING SUMMARY")
    logging.info("="*50)
    logging.info(f"Total images: {stats['total']}")
    logging.info(f"Images with faces: {stats['with_faces']}")
    logging.info(f"Images without faces: {stats['no_faces']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info(f"Output saved to: {out_dir}")
    logging.info("="*50)


if __name__ == '__main__':
    main()