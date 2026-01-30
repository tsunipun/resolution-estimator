import gradio as gr
import torch
import os
from PIL import Image
from torchvision import transforms
from model import ResolutionEstimator

# Global variables
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path='checkpoints/best_model.pth'):
    global model
    print(f"Loading model from {model_path}...")
    model = ResolutionEstimator(pretrained=False)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Using random weights instead.")
    else:
        print("Model checkoint not found. Using random weights.")
    
    model.to(device)
    model.eval()

def predict(image):
    if image is None:
        return None, "No image", "N/A"
    
    try:
        # Convert to PIL if necessary (Gradio passes numpy or PIL depending on type)
        # We'll stick to 'pil' type in Interface
        w, h = image.size
        
        # Crop Size
        crop_size = 224
        
        # Pad if needed
        if w < crop_size or h < crop_size:
            pad_w = max(0, crop_size - w)
            pad_h = max(0, crop_size - h)
            img_padded = transforms.functional.pad(image, (0, 0, pad_w, pad_h))
            w_curr, h_curr = img_padded.size
        else:
            img_padded = image
            w_curr, h_curr = w, h

        # 5-Crop Strategy
        crops = []
        crops.append(transforms.functional.center_crop(img_padded, (crop_size, crop_size)))
        crops.append(transforms.functional.crop(img_padded, 0, 0, crop_size, crop_size)) # TL
        crops.append(transforms.functional.crop(img_padded, 0, w_curr - crop_size, crop_size, crop_size)) # TR
        crops.append(transforms.functional.crop(img_padded, h_curr - crop_size, 0, crop_size, crop_size)) # BL
        crops.append(transforms.functional.crop(img_padded, h_curr - crop_size, w_curr - crop_size, crop_size, crop_size)) # BR
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        batch = torch.stack([transform(c) for c in crops]).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            score = outputs.mean().item()
            
        # Results
        score_display = f"{score:.4f}"
        
        est_w = int(w * score)
        est_h = int(h * score)
        
        orig_res = f"{w} x {h}"
        est_res = f"{est_w} x {est_h}"
        
        return score_display, orig_res, est_res

    except Exception as e:
        return "Error", str(e), "N/A"

# Initialize Model
load_model()

# Gradio Interface
with gr.Blocks(title="Resolution Estimator") as demo:
    gr.Markdown("# Image Resolution Estimator\nUpload an image to estimate its 'true' resolution quality (ignoring upscaling).")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Input Image")
            run_btn = gr.Button("Estimate Resolution", variant="primary")
        
        with gr.Column():
            score_out = gr.Label(label="Quality Score (0.0 - 1.0)")
            orig_res_out = gr.Textbox(label="Current Resolution")
            est_res_out = gr.Textbox(label="Estimated True Resolution")
    
    run_btn.click(predict, inputs=[input_img], outputs=[score_out, orig_res_out, est_res_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
