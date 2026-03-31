import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Import the custom classes we made in the other files
from model import UNET, DiceLoss
from dataset import DrivableDataset

# --- HYPERPARAMETERS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
BATCH_SIZE = 4 
NUM_EPOCHS = 40
IMAGE_HEIGHT, IMAGE_WIDTH = 640, 640
TRAIN_IMG_DIR = "data1/train/"
TRAIN_MASK_DIR = "data1/train_masks/"

def train_fn(loader, model, optimizer, bce_loss, dice_loss, device):
    """
    Performs one epoch of training.
    """
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        # Forward pass
        predictions = model(data)
        
        # Hybrid Loss (BCE + Dice)
        loss = bce_loss(predictions, targets) + dice_loss(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm progress bar
        loop.set_postfix(loss=loss.item())

def main():
    # 1. Transforms (Data Augmentation for Level 4 robustness)
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.Rotate(limit=35, p=0.2),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # 2. Dataset & Loader
    train_ds = DrivableDataset(
        image_dir=TRAIN_IMG_DIR, 
        mask_dir=TRAIN_MASK_DIR, 
        transform=train_transform
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2, 
        pin_memory=True
    )

    # 3. Model, Loss, Optimizer
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, bce_loss, dice_loss, DEVICE)
        
        # Save Model Checkpoint
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, "my_checkpoint.pth")
        print("Checkpoint saved!")

if __name__ == "__main__":
    main()