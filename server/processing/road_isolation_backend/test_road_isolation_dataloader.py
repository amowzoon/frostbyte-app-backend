import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

# Import your dataset
from road_isolation_dataloader import AcclimateWeatherDataset  # adjust path as needed


def test_dataset_structure(dataset_root):
    """Verify directory and file structure."""
    weather_types = ["fog", "night", "rain", "snow"]
    expected_subfolders = ["train", "val", "test"]

    for weather in weather_types:
        rgb_path = os.path.join(dataset_root, "rgb_images", weather)
        gt_path = os.path.join(dataset_root, "gt", weather)

        assert os.path.exists(rgb_path), f"Missing folder: {rgb_path}"
        assert os.path.exists(gt_path), f"Missing folder: {gt_path}"

        for sub in expected_subfolders:
            rgb_sub = os.path.join(rgb_path, sub)
            gt_sub = os.path.join(gt_path, sub)
            assert os.path.exists(rgb_sub), f"Missing {rgb_sub}"
            # ground truth for test is not available, must use their website
            if sub != "test":
                assert os.path.exists(gt_sub), f"Missing {gt_sub}"

    print("✅ Dataset structure OK.")


def test_sample_alignment(dataset):
    """Check that each RGB frame has a corresponding GT."""
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, dict)
        assert "image" in sample and "mask" in sample, "Sample missing keys"
        img, mask = sample["image"], sample["mask"]

        # Torch tensor sanity
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

        # Matching dimensions
        
        # expand to ensure that we account for RGB dimension
        
        mask = mask.unsqueeze(0)
        assert img.shape[1:] == mask.shape[1:], f"Mismatched image/mask at index {i}"
        if i < 5:
            print(f"✅ Sample {i}: shape {img.shape}, dtype {img.dtype}")
    print("✅ Sample alignment OK.")


def test_dataloader_visual(dataset, num_samples=4):
    """Visualize a few samples."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for i, sample in enumerate(loader):
        img = sample["image"][0].permute(1, 2, 0).numpy()
        mask = sample["mask"][0].numpy()  # assuming single-channel mask

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img.astype(np.uint8))
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Mask")
        plt.show()

        if i + 1 >= num_samples:
            break
    print("✅ Visualization test completed.")


def run_all_tests():
    root = "/projectnb/frostbyte/Datasets/acdc_acclimate_weather"
    dataset = AcclimateWeatherDataset(
        root_dir=root,
        split="train",
	transform=None,
        weather_list=None
    )

    test_dataset_structure(root)
    test_sample_alignment(dataset)
    test_dataloader_visual(dataset)


if __name__ == "__main__":
    run_all_tests()
