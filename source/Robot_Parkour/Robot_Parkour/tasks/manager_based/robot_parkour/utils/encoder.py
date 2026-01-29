import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import time



class DepthEncoder(nn.Module):
    def __init__(self, input_ch=1, hidden_dims=128):
        super().__init__()

        # Encoder Specification
        # Input: (1, 64, 64)
        self.conv_blocks = nn.Sequential(
            # Layer 1: 1 -> 16, Kernel 5, Stride 2
            nn.Conv2d(input_ch, 16, kernel_size=5, stride=2, padding=2), # 64 -> 32
            nn.ReLU(),
            # Optional: nn.MaxPool2d(2) -- User mentioned pooling, but stride=2 is already downsampling

            # Layer 2: 16 -> 32, Kernel 4, Stride 1
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1), # 32 -> 31 (approx, padding=1 keeps it close)
            nn.ReLU(),

            # Layer 3: 32 -> 32, Kernel 3, Stride 1
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Calculate Flatten size automatically
        # Assuming input 64x64 and the padding above:
        # L1: 32x32 -> L2: 31x31 -> L3: 31x31 (roughly).
        # To be safe, we use a dummy pass to find the size.
        with torch.no_grad():
            dummy = torch.zeros(1, input_ch, 64, 64)
            out = self.conv_blocks(dummy)
            self.flatten_dim = out.numel()

        self.fc = nn.Linear(self.flatten_dim, hidden_dims)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

class DepthDecoder(nn.Module):
    def __init__(self, embedded_dim=128, output_ch=1):
        super().__init__()
        # We need to reverse the mapping.
        # Linear -> Unflatten -> Transposed Convs

        # This size (32 * 31 * 31) depends on the exact encoder output
        # For simplicity in this script, we will project back to a manageable feature map (e.g., 32x8x8)
        # and upsample from there to 64x64.
        self.init_h = 8
        self.init_w = 8
        self.init_ch = 32

        self.fc = nn.Linear(embedded_dim, self.init_ch * self.init_h * self.init_w)

        self.deconv_blocks = nn.Sequential(
            # Unflattened input: (32, 8, 8)

            # Layer 1: Upsample 8 -> 16
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Layer 2: Upsample 16 -> 32
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            # Layer 3: Upsample 32 -> 64
            nn.ConvTranspose2d(16, output_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Normalize output to [0, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.init_ch, self.init_h, self.init_w)
        x = self.deconv_blocks(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DepthEncoder()
        self.decoder = DepthDecoder()

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# ==========================================
# Dataset Loader
# ==========================================

class DepthDataset(Dataset):
    def __init__(self, data_path):
        # Assumes data is saved as a .npy or .pt tensor of shape (N, 64, 64)
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path) # or np.load(data_path)

        # normalize input to [0, 1]
        self.data = self.data / torch.max(self.data)

        # Ensure shape is (N, 1, 64, 64)
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(1)

        print(f"Data loaded: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return depth image.
        # Data should already be normalized [0,1] or handle it here.
        return self.data[idx].float()





def train():
    # Settings
    DATA_PATH = "/home/luca/dev/Robot_Parkour/datasets/depth_dataset.pt"
    BATCH_SIZE = 256
    LR = 1e-3
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Check if data exists
    if not os.path.exists(DATA_PATH):
        print("Dataset not found! Please run the dataset generation snippet first.")
        return

    # Setup
    dataset = DepthDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = AutoEncoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Reconstruction loss

    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()

        for batch in dataloader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            recon, latent = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Time: {time.time()-start_time:.1f}s")

    # Save the ENCODER only (this is what you need for the student)
    torch.save(model.state_dict(), "datasets/depth_encoder.pt")
    print("Training complete. Encoder saved to 'depth_encoder.pt'")


def visualize():
    # SETTINGS
    MODEL_PATH = "datasets/depth_encoder.pt"
    DATA_PATH = "datasets/depth_dataset.pt"
    NUM_SAMPLES = 5  # How many pairs to show

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print("Loading dataset...")
    data = torch.load(DATA_PATH)

    max_val = torch.max(data)
    print(f"Normalizing data using max value: {max_val:.4f}")
    data = data / max_val

    # Ensure shape (N, 1, 64, 64)
    if len(data.shape) == 3:
        data = data.unsqueeze(1)

    # 2. Load Model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found. Did you save the full model state_dict?")
        return

    model = AutoEncoder().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. Select Random Samples
    indices = random.sample(range(len(data)), NUM_SAMPLES)
    samples = data[indices].to(device)

    # 4. Run Inference
    with torch.no_grad():
        reconstructions, _ = model(samples)

    # 5. Plot Side-by-Side
    # Create a figure with 2 columns and NUM_SAMPLES rows
    fig, axes = plt.subplots(nrows=NUM_SAMPLES, ncols=2, figsize=(6, 3 * NUM_SAMPLES))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    samples = samples.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    for i in range(NUM_SAMPLES):
        # Original
        ax_orig = axes[i, 0]
        # Squeeze to remove channel dim: (1, 64, 64) -> (64, 64)
        ax_orig.imshow(samples[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_orig.axis('off')
        if i == 0: ax_orig.set_title("Input (Depth)", fontsize=10)

        # Reconstructed
        ax_recon = axes[i, 1]
        ax_recon.imshow(reconstructions[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        ax_recon.axis('off')
        if i == 0: ax_recon.set_title("Decoder Output", fontsize=10)

    print("Displaying plot...")
    plt.show()



if __name__ == "__main__":
    train()