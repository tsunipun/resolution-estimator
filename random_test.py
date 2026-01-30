import argparse
import os
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from model import ResolutionEstimator
from tqdm import tqdm

def predict_score(model, img, device, crop_size=224):
    """
    Predicts resolution score using 5-crop average strategy.
    """
    w, h = img.size
    
    # Pad if smaller than crop size
    if w < crop_size or h < crop_size:
        pad_w = max(0, crop_size - w)
        pad_h = max(0, crop_size - h)
        img = transforms.functional.pad(img, (0, 0, pad_w, pad_h))
        w_curr, h_curr = img.size
    else:
        w_curr, h_curr = w, h

    crops = []
    # Center
    crops.append(transforms.functional.center_crop(img, (crop_size, crop_size)))
    # Corners
    crops.append(transforms.functional.crop(img, 0, 0, crop_size, crop_size)) # TL
    crops.append(transforms.functional.crop(img, 0, w_curr - crop_size, crop_size, crop_size)) # TR
    crops.append(transforms.functional.crop(img, h_curr - crop_size, 0, crop_size, crop_size)) # BL
    crops.append(transforms.functional.crop(img, h_curr - crop_size, w_curr - crop_size, crop_size, crop_size)) # BR
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    batch = torch.stack([transform(c) for c in crops]).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        score = outputs.mean().item()
        
    return score

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
        print(f"Warning: Model path {args.model_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Get images
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    else:
        print(f"Input directory {args.input} not found or is not a directory.")
        return

    if not files:
        print("No images found in input directory.")
        return

    print(f"Found {len(files)} images. Starting random degradation test with 10 variations per image...")
    
    errors = []
    
    for f in tqdm(files):
        try:
            original = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

        w, h = original.size
        
        # Test 10 random scales for each image
        for _ in range(10):
            # 1. Random Scale Factor (Label)
            s = random.uniform(0.1, 1.0)
            
            # 2. Degrade
            new_w = max(1, int(w * s))
            new_h = max(1, int(h * s))
            
            # Resize down
            img_deg = original.resize((new_w, new_h), resample=Image.BICUBIC)
            # Resize up
            img_restored = img_deg.resize((w, h), resample=Image.BICUBIC)
            
            # 3. Predict
            predicted_s = predict_score(model, img_restored, device)
            
            # 4. Error
            error = abs(predicted_s - s)
            errors.append(error)
        
            # Optional: Print detailed debug occasionally or if error is large?
            # tqdm.write(f"{os.path.basename(f)}: True={s:.3f}, Pred={predicted_s:.3f}, Err={error:.3f}")

    if errors:
        mae = np.mean(errors)
        std_err = np.std(errors)
        print("\n" + "="*30)
        print(f"Test Completed.")
        print(f"Images Processed: {len(files)}")
        print(f"Total Predictions: {len(errors)}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Error Std Dev: {std_err:.4f}")
        print("="*30)
    else:
        print("No successful predictions made.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random Resolution Degradation Test')
    parser.add_argument('--input', type=str, default='test_images', help='Directory of test images')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to trained model')
    
    args = parser.parse_args()
    main(args)
