import torch
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# Force Matplotlib to not pop up windows (Prevents the "Endless Loop" lag)
matplotlib.use('Agg')

# Import your custom architecture from your local model.py
from model import UNET 
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- CONFIGURATION ---
# NOTE: If you don't have an NVIDIA GPU on your laptop, this will default to "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "my_checkpoint.pth"
TEST_IMG_DIR = "data1/test/"
TEST_MASK_DIR = "data1/test_masks/"
OUTPUT_FOLDER = "test_results/"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

test_transform = A.Compose([
    A.Resize(height=640, width=640),
    A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
    ToTensorV2(),
])

def run_evaluation():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found!")
        return

    print(f"Loading weights from {CHECKPOINT_PATH}...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Logic to handle both "Direct" and "Wrapped" checkpoints
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    all_iou = []
    inference_times = []
    images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Processing {len(images)} images on {DEVICE.upper()}...")

    # tqdm shows a progress bar so you know it's not stuck
    for img_name in tqdm(images, desc="Generating Predictions"):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        mask_path = os.path.join(TEST_MASK_DIR, img_name)
        
        if not os.path.exists(mask_path):
            continue

        image_raw = np.array(Image.open(img_path).convert("RGB"))
        image_tensor = test_transform(image=image_raw)["image"].unsqueeze(0).to(DEVICE)
        
        ground_truth = np.array(Image.open(mask_path).convert("L"))
        ground_truth = (ground_truth > 128).astype(np.float32)

        # --- INFERENCE ---
        start_tick = time.time()
        with torch.no_grad():
            # AMP only works on GPU; we skip it on CPU to avoid errors
            if DEVICE == "cuda":
                with torch.cuda.amp.autocast():
                    prediction = model(image_tensor)
            else:
                prediction = model(image_tensor)
                
            prediction = torch.sigmoid(prediction)
            prediction = (prediction > 0.5).float().squeeze().cpu().numpy()
        
        end_tick = time.time()
        inference_times.append(end_tick - start_tick)

        # --- IoU MATH ---
        intersection = np.logical_and(ground_truth, prediction).sum()
        union = np.logical_or(ground_truth, prediction).sum()
        all_iou.append(intersection / union if union > 0 else 1.0)

        # --- SAVE PLOT QUIETLY ---
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plt.imshow(image_raw); plt.axis("off")
        plt.subplot(1, 2, 2); plt.imshow(prediction, cmap='magma'); plt.axis("off")
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"res_{img_name}"), bbox_inches='tight')
        plt.close('all') # Clears memory after every image

    avg_miou = np.mean(all_iou) * 100
    avg_fps = 1 / np.mean(inference_times)
    
    print("\n" + "="*40)
    print("      FINAL VALIDATION REPORT")
    print("="*40)
    print(f"Mean IoU (mIoU):    {avg_miou:.2f}%")
    print(f"Inference Speed:    {avg_fps:.2f} FPS")
    print(f"Device Used:        {DEVICE.upper()}")
    print(f"Results saved to:    {OUTPUT_FOLDER}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()