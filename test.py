import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import ResolutionEstimator

def predict_image(model, image_path, device, crop_size=224):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        return None

    w, h = img.size
    
    # If image is smaller than crop, just resize/pad?
    # For inference, if it's smaller than crop, it's definitely low res in absolute terms.
    # But let's handle it by padding.
    if w < crop_size or h < crop_size:
        pad_w = max(0, crop_size - w)
        pad_h = max(0, crop_size - h)
        img = transforms.functional.pad(img, (0, 0, pad_w, pad_h))
        w_curr, h_curr = img.size
    else:
        w_curr, h_curr = w, h

    # Strategy: Take 5 crops (Center, TopLeft, TopRight, BottomLeft, BottomRight)
    # and average the predictions.
    
    crops = []
    
    # Center
    crops.append(transforms.functional.center_crop(img, (crop_size, crop_size)))
    
    # Corners
    crops.append(transforms.functional.crop(img, 0, 0, crop_size, crop_size)) # TL
    crops.append(transforms.functional.crop(img, 0, w_curr - crop_size, crop_size, crop_size)) # TR
    crops.append(transforms.functional.crop(img, h_curr - crop_size, 0, crop_size, crop_size)) # BL
    crops.append(transforms.functional.crop(img, h_curr - crop_size, w_curr - crop_size, crop_size, crop_size)) # BR
    
    # Pre-define transform (constant for all images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    batch = torch.stack([transform(c) for c in crops]).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        # avg score
        score = outputs.mean().item()
        
    # Standard deviation to check consistency?
    std = outputs.std().item()
    
    return score, w, h, std

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = ResolutionEstimator(pretrained=False)
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    else:
        print("Model path not found! Using random weights (for testing only).")
    
    model.to(device)
    model.eval()
    
    # Process
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    else:
        print("Invalid input path.")
        return
        
    from tqdm import tqdm
    print(f"Processing {len(files)} images...")
    
    for f in tqdm(files):
        result = predict_image(model, f, device)
        if result:
            score, orig_w, orig_h, std = result
            
            # Estimated "True" Resolution
            est_w = int(orig_w * score)
            est_h = int(orig_h * score)
            
            tqdm.write(f"Image: {os.path.basename(f)}")
            tqdm.write(f"  Current Size: {orig_w}x{orig_h}")
            tqdm.write(f"  Quality Score: {score:.4f} (std: {std:.4f})")
            tqdm.write(f"  Estimated True Size: {est_w}x{est_h}")
            tqdm.write("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Resolution Estimator')
    parser.add_argument('--input', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model')
    
    args = parser.parse_args()
    
    # Optimization: Helper for faster inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    main(args)
