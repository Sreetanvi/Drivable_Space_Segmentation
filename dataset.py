import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DrivableDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Only load valid image formats
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        image = np.array(Image.open(img_path).convert("RGB"))
        # NuScenes masks are usually grayscale; convert to binary 0/1
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 128).astype(np.float32)

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask