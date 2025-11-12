"""
Extract global scene features using ResNet50/DenseNet161
Saves whole-image features as .npz (scene: 1×D)
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

import torchvision.models as models
from torchvision import transforms


def setup_logging(out_dir):
    log_file = os.path.join(out_dir, 'process_scene.log')
    error_file = os.path.join(out_dir, 'errors_scene.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return error_file


def load_scene_model(model_name='resnet50', device='cuda'):
    """
    Load pretrained CNN for scene feature extraction
    Options: resnet50, densenet161
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        feat_dim = 2048
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.classifier = torch.nn.Identity()
        feat_dim = 2208
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return model, transform, feat_dim


def extract_scene_feature(img_rgb, model, transform, device):
    """Extract global scene feature from whole image"""
    img_pil = Image.fromarray(img_rgb)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat = model(img_tensor)
        feat = feat.squeeze().cpu().numpy()
    
    return feat


def process_image(img_path, model, transform, device, error_file):
    """Process image and extract scene feature"""
    try:
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            error_msg = f"Failed to read: {img_path}"
            logging.error(error_msg)
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(error_msg + '\n')
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract scene feature
        feature = extract_scene_feature(img_rgb, model, transform, device)
        
        return feature.reshape(1, -1)  # Shape: (1, D)
        
    except Exception as e:
        error_msg = f"Error processing {img_path}: {str(e)}"
        logging.error(error_msg)
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract scene features')
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'densenet161'])
    
    args = parser.parse_args()
    
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
    
    # Load model
    logging.info(f"Loading {args.model}...")
    model, transform, feat_dim = load_scene_model(args.model, device)
    logging.info(f"Feature dim: {feat_dim}")
    
    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    logging.info(f"Processing {len(image_files)} images")
    
    stats = {'total': len(image_files), 'success': 0, 'errors': 0}
    
    for img_path in tqdm(image_files, desc="Extracting scenes"):
        feature = process_image(img_path, model, transform, device, error_file)
        
        if feature is None:
            stats['errors'] += 1
            continue
        
        out_file = out_dir / f"{img_path.stem}.npz"
        np.savez_compressed(out_file, scene=feature)
        stats['success'] += 1
    
    logging.info("\n" + "="*50)
    logging.info("SCENE EXTRACTION SUMMARY")
    logging.info("="*50)
    logging.info(f"Total: {stats['total']}")
    logging.info(f"Success: {stats['success']}")
    logging.info(f"Errors: {stats['errors']}")
    logging.info("="*50)


if __name__ == '__main__':
    main()