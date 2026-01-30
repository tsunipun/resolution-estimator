import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ResolutionDataset(Dataset):
    def __init__(self, root_dir, crop_size=224, transform=None):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.image_files = []
        
        # specific extensions
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # Walk through directory
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(exts):
                    self.image_files.append(os.path.join(root, file))
        
        # Base transform for normalization
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If load fails, return a random other image or handle error
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 1. Determine random scale factor label
        # Range [0.1, 1.0]. 
        # Bias towards 1.0? Or uniform? Uniform is fine.
        s = random.uniform(0.1, 1.0)
        
        # 2. Degrade
        w, h = image.size
        
        # If image is too small for crop, we might not want to downscale it purely
        # But we must satisfy the crop size.
        # If w < crop_size or h < crop_size, we should probably resize the ORIGINAL up first
        # so we have enough canvas. But this adds blur!
        # Better strategy: Filter dataset for images >= crop_size.
        # If we encounter small image, skip it (handled in __init__ ideally, but expensive)
        # We'll just pad if needed.
        
        # Logic: 
        # Downscale
        new_w = max(1, int(w * s))
        new_h = max(1, int(h * s))
        
        # Resize down (degradation)
        img_deg = image.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Resize up (restore dimensions)
        # Randomly choose interpolation method to expose model to different artifacts
        resample_method = random.choice([
            Image.NEAREST, 
            Image.BILINEAR, 
            Image.BICUBIC, 
            Image.LANCZOS, 
            Image.BOX, 
            Image.HAMMING
        ])
        img_restored = img_deg.resize((w, h), resample=resample_method)
        
        # 3. Random Crop to fixed size
        if w < self.crop_size or h < self.crop_size:
            # Pad to fit crop_size
            pad_w = max(0, self.crop_size - w)
            pad_h = max(0, self.crop_size - h)
            img_restored = transforms.functional.pad(img_restored, (0, 0, pad_w, pad_h))
            # Recalculate size
            w, h = img_restored.size
            
        i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
            img_restored, output_size=(self.crop_size, self.crop_size))
            
        crop = transforms.functional.crop(img_restored, i, j, h_crop, w_crop)
        
        # 4. Augmentation (flip/rotate) - only affecting visual, not resolution label
        if random.random() > 0.5:
            crop = transforms.functional.hflip(crop)
        
        # 5. To Tensor
        data = self.base_transform(crop)
        
        # Label is s (float)
        label = torch.tensor([s], dtype=torch.float32)
        
        return data, label
