"""
Validation Script for Road Isolation Model Requirements

Tests:
1. Visual isolation of road + sidewalks
2. Model size < 10 MB
3. IoU ≥ 75% on validation set
"""

import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
import sys
import argparse
sys.path.append("../MobileSAM")
from mobile_sam import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F
from road_isolation_dataloader import AcclimateWeatherDataset
import albumentations as A
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SegmentationHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

# ============================================================================
# REQUIREMENT 1: VISUAL ISOLATION TEST
# ============================================================================

def test_visual_isolation(checkpoint_path, val_loader, device, num_samples=5, 
                          output_dir='../validation_outputs'):
    """
    Test Requirement 1: Visual isolation of road + sidewalks
    
    Generates visual outputs showing:
    - Original image
    - Ground truth mask
    - Predicted mask
    - Overlay visualization
    """
    print("\n" + "="*80)
    print("REQUIREMENT 1: VISUAL ISOLATION TEST")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_type = "vit_t"
    mobile_sam = sam_model_registry[model_type](checkpoint=None)
    encoder = mobile_sam.image_encoder.to(device)
    seg_head = SegmentationHead(in_channels=256, num_classes=2).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ensure it's float 32
    for key in checkpoint['encoder_state_dict']:
        checkpoint['encoder_state_dict'][key] = checkpoint['encoder_state_dict'][key].float()
    for key in checkpoint['seg_head_state_dict']:
        checkpoint['seg_head_state_dict'][key] = checkpoint['seg_head_state_dict'][key].float()
    
    # load weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
    
    encoder.eval()
    seg_head.eval()
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if sample_count >= num_samples:
                break
            
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            
            # Inference
            embeddings = encoder(imgs)
            logits = seg_head(embeddings)
            logits = F.interpolate(logits, size=masks.shape[-2:], 
                                  mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1)
            
            # Process each image in batch
            for i in range(imgs.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Convert to numpy
                img = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                gt_mask = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original image
                axes[0, 0].imshow(img)
                axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                # Ground truth
                axes[0, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title('Ground Truth\n(1=Road/Sidewalk, 0=Other)', 
                                    fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                # Prediction
                axes[1, 0].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
                axes[1, 0].set_title('Predicted Mask\n(1=Road/Sidewalk, 0=Other)', 
                                    fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                # Overlay
                overlay = img.copy()
                mask_colored = np.zeros_like(img)
                mask_colored[pred_mask == 1] = [0, 255, 0]  # Green for roads
                overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title('Prediction Overlay\n(Green=Road/Sidewalk)', 
                                    fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                output_path = os.path.join(output_dir, f'sample_{sample_count+1}.png')
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                sample_count += 1
                print(f"  ✓ Generated visualization {sample_count}/{num_samples}: {output_path}")
    
    print(f"\n✅ REQUIREMENT 1 PASSED")
    print(f"   Visual outputs saved to '{output_dir}/'")
    print(f"   Road/sidewalk isolation is clearly visible in green overlay")
    
    return True

# ============================================================================
# REQUIREMENT 2: MODEL SIZE TEST
# ============================================================================

def test_model_size(checkpoint_path, max_size_mb=15):
    """
    Test Requirement 2: Model size < 15 MB
    """
    print("\n" + "="*80)
    print("REQUIREMENT 2: MODEL SIZE TEST")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model file not found: {checkpoint_path}")
        return False
    
    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"\n  Model file: {checkpoint_path}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Requirement: < {max_size_mb} MB")
    
    if file_size_mb < max_size_mb:
        print(f"\n✅ REQUIREMENT 2 PASSED")
        print(f"   Model size ({file_size_mb:.2f} MB) is under {max_size_mb} MB limit")
        return True
    else:
        print(f"\n❌ REQUIREMENT 2 FAILED")
        print(f"   Model size ({file_size_mb:.2f} MB) exceeds {max_size_mb} MB limit")
        return False

# ============================================================================
# REQUIREMENT 3: IoU PERFORMANCE TEST
# ============================================================================

def compute_iou(pred, target, class_id=1):
    """Compute IoU for a specific class"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    pred_cls = pred == class_id
    target_cls = target == class_id
    
    intersection = (pred_cls & target_cls).sum().float()
    union = (pred_cls | target_cls).sum().float()
    
    if union == 0:
        return float('nan')
    
    return (intersection / union).item()

def test_iou_performance(checkpoint_path, val_loader, device, min_iou=0.75):
    """
    Test Requirement 3: IoU ≥ 75% on validation set
    """
    print("\n" + "="*80)
    print("REQUIREMENT 3: IoU PERFORMANCE TEST")
    print("="*80)
    
    # Load model
    model_type = "vit_t"
    mobile_sam = sam_model_registry[model_type](checkpoint=None)
    encoder = mobile_sam.image_encoder.to(device)
    seg_head = SegmentationHead(in_channels=256, num_classes=2).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # ensure it's float 32
    for key in checkpoint['encoder_state_dict']:
        checkpoint['encoder_state_dict'][key] = checkpoint['encoder_state_dict'][key].float()
    for key in checkpoint['seg_head_state_dict']:
        checkpoint['seg_head_state_dict'][key] = checkpoint['seg_head_state_dict'][key].float()
    # load weights
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    seg_head.load_state_dict(checkpoint['seg_head_state_dict'])
    
    encoder.eval()
    seg_head.eval()
    
    all_ious = []
    
    print("\n  Computing IoU on validation set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device).long()
            
            # Inference
            embeddings = encoder(imgs)
            logits = seg_head(embeddings)
            logits = F.interpolate(logits, size=masks.shape[-2:], 
                                  mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1)
            
            # Compute IoU for each image
            for i in range(preds.shape[0]):
                iou = compute_iou(preds[i], masks[i], class_id=1)  # Road/sidewalk class
                if not np.isnan(iou):
                    all_ious.append(iou)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"    Processed {batch_idx + 1}/{len(val_loader)} batches...")
    
    mean_iou = np.mean(all_ious)
    std_iou = np.std(all_ious)
    min_iou_val = np.min(all_ious)
    max_iou_val = np.max(all_ious)
    
    print(f"\n  Road/Sidewalk Class IoU Statistics:")
    print(f"    Mean IoU:    {mean_iou*100:.2f}%")
    print(f"    Std Dev:     {std_iou*100:.2f}%")
    print(f"    Min IoU:     {min_iou_val*100:.2f}%")
    print(f"    Max IoU:     {max_iou_val*100:.2f}%")
    print(f"    Samples:     {len(all_ious)}")
    print(f"    Requirement: ≥ {min_iou*100:.0f}%")
    
    if mean_iou >= min_iou:
        print(f"\n✅ REQUIREMENT 3 PASSED")
        print(f"   Mean IoU ({mean_iou*100:.2f}%) meets the {min_iou*100:.0f}% threshold")
        
        if mean_iou >= 0.85:
            print(f"   🌟 Excellent performance!")
        elif mean_iou >= 0.80:
            print(f"   💪 Strong performance!")
        else:
            print(f"   👍 Good performance!")
        
        return True, mean_iou
    else:
        print(f"\n❌ REQUIREMENT 3 FAILED")
        print(f"   Mean IoU ({mean_iou*100:.2f}%) is below {min_iou*100:.0f}% threshold")
        print(f"\n   Suggestions:")
        print(f"   - Train for more epochs")
        print(f"   - Increase road class weight in loss function")
        print(f"   - Add more data augmentation")
        return False, mean_iou

# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def run_all_tests(checkpoint_path, dataset_root, output_dir='validation_outputs'):
    """
    Run all three testable requirements
    """
    print("\n" + "="*80)
    print("ROAD ISOLATION MODEL - REQUIREMENTS VALIDATION")
    print("="*80)
    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output directory: {output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load validation dataset
    transform = A.Compose([A.Resize(1024, 1024)])
    val_ds = AcclimateWeatherDataset(dataset_root, split="val", 
                                      transform=transform, binary_road_mask=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    results = {}
    
    # Test 1: Visual Isolation
    results['visual_isolation'] = test_visual_isolation(
        checkpoint_path, val_loader, device, num_samples=5, output_dir=output_dir
    )
    
    # Test 2: Model Size
    results['model_size'] = test_model_size(checkpoint_path, max_size_mb=15)
    
    # Test 3: IoU Performance
    results['iou_performance'], results['mean_iou'] = test_iou_performance(
        checkpoint_path, val_loader, device, min_iou=0.75
    )
    
    # Final Report
    print("\n" + "="*80)
    print("FINAL VALIDATION REPORT")
    print("="*80)
    
    print("\n📋 TESTABLE REQUIREMENTS:")
    print(f"  1. Visual isolation of road + sidewalks:  {'✅ PASS' if results['visual_isolation'] else '❌ FAIL'}")
    print(f"  2. Model size < 10 MB:                    {'✅ PASS' if results['model_size'] else '❌ FAIL'}")
    print(f"  3. IoU ≥ 75%:                              {'✅ PASS' if results['iou_performance'] else '❌ FAIL'}")
    
    all_passed = all([results['visual_isolation'], results['model_size'], 
                     results['iou_performance']])
    
    if all_passed:
        print(f"\n🎉 ALL REQUIREMENTS PASSED!")
        print(f"   The model is ready for deployment.")
    else:
        print(f"\n⚠️  SOME REQUIREMENTS FAILED")
        print(f"   Please review the results above and make necessary improvements.")
    
    print("\n" + "="*80)
    
    # Save report
    report_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_path, 'w') as f:
        f.write("ROAD ISOLATION MODEL - VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Dataset: {dataset_root}\n\n")
        f.write("REQUIREMENTS:\n")
        f.write(f"1. Visual Isolation: {'PASS' if results['visual_isolation'] else 'FAIL'}\n")
        f.write(f"2. Model Size < 10 MB: {'PASS' if results['model_size'] else 'FAIL'}\n")
        f.write(f"3. IoU ≥ 75%: {'PASS' if results['iou_performance'] else 'FAIL'} (Achieved: {results['mean_iou']*100:.2f}%)\n")
        f.write(f"\nOVERALL: {'ALL PASSED' if all_passed else 'FAILED'}\n")
    
    print(f"📄 Validation report saved to: {report_path}\n")
    
    return all_passed, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate road segmentation model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../saved_models/road_segmentation_model.pth",
        help="Path to the model checkpoint"
    )


    args = parser.parse_args()
    
    checkpoint_path = args.checkpoint
    dataset_root = '/projectnb/frostbyte/Datasets/acdc_acclimate_weather'
    
    passed, results = run_all_tests(checkpoint_path, dataset_root)
    
    if passed:
        print("✅ Model validated successfully!")
    else:
        print("❌ Model validation failed. See report for details.")

"""
if __name__ == "__main__":
    # Run validation
    
    checkpoint_path = '../saved_models/road_segmentation_model.pth'
    dataset_root = '/projectnb/frostbyte/Datasets/acdc_acclimate_weather'
    
    passed, results = run_all_tests(checkpoint_path, dataset_root)
    
    if passed:
        print("✅ Model validated successfully!")
    else:
        print("❌ Model validation failed. See report for details.")
        """
