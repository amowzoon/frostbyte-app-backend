import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

class AcclimateWeatherDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, weather_list=None, 
                 binary_road_mask=True):
        """
        Args:
            root_dir: e.g. /projectnb/frostbyte/Datasets/acdc_acclimate_weather
            split: "train", "val", or "test"
            transform: Albumentations pipeline (optional)
            weather_list: list of weathers to include, e.g. ["fog", "snow"]. Defaults to all.
            binary_road_mask: If True, remap to binary (road/sidewalk=1, else=0)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.binary_road_mask = binary_road_mask
        self.samples = []
        
        rgb_root = os.path.join(root_dir, "rgb_images")
        gt_root = os.path.join(root_dir, "gt")
        
        if weather_list is None:
            weather_list = [w for w in os.listdir(rgb_root) 
                          if os.path.isdir(os.path.join(rgb_root, w))]
        
        for weather in weather_list:
            rgb_weather = os.path.join(rgb_root, weather, split)
            gt_weather = os.path.join(gt_root, weather, split)
            
            if not os.path.isdir(rgb_weather) or not os.path.isdir(gt_weather):
                continue
            
            # Recursively find all RGB frames
            rgb_files = glob.glob(os.path.join(rgb_weather, "**", "*_rgb_anon.png"), 
                                recursive=True)
            
            for rgb_path in rgb_files:
                rel_path = os.path.relpath(rgb_path, rgb_weather)
                gt_path = os.path.join(gt_weather, 
                                      rel_path.replace("_rgb_anon.png", "_gt_labelTrainIds.png"))
                
                if os.path.exists(gt_path):
                    self.samples.append((rgb_path, gt_path))
                else:
                    if split != "test":
                        print(f"Missing GT mask: {gt_path}")
        
        print(f"[{split}] Found {len(self.samples)} images across weathers {weather_list}")
        if self.binary_road_mask:
            print(f"[{split}] Binary mask mode: road(0) + sidewalk(1) -> class 1, rest -> class 0")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]
        
        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        
        # Convert to tensors
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)
        
        # Remap mask to binary if enabled
        if self.binary_road_mask:
            # ACDC classes: 0=road, 1=sidewalk, 2-18=other, 255=void
            # Map to: 1=road/sidewalk, 0=everything else
            binary_mask = torch.zeros_like(mask)
            binary_mask[(mask == 0) | (mask == 1)] = 1
            mask = binary_mask
        
        return {"image": image, "mask": mask}
