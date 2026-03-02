"""
Load a pretrained model and run inference on it.
For the double-encoder model
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from double_train_fm import ConditionalFlowMatchingModule
# from train_fm import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader, TensorDataset
from data import HSCLegacyDataset
import time

import umap
import matplotlib.pyplot as plt
import numpy as np

# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/epdlfvpg/checkpoints/epoch=123-step=46000.ckpt'
# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/33mo9r3n/checkpoints/epoch=201-step=75000.ckpt' # z_dim = 512

# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/22teteus/checkpoints/epoch=18-step=7000.ckpt'

checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/s39qr0v8/checkpoints/epoch=201-step=75000.ckpt' # model with latent space of 128

# Dimension setting (used in plot file names)
dim = 128  # Set to 32, 64, 128, 256, 512, etc.

# Control flags
GENERATE_UMAP = True  # Set to False to skip UMAP generation and plotting
SHOW_PAIRS = True      # Set to False to skip marking pairs on the plots
GENERATE_SAMPLES = True  # Set to False to skip generation study

model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path)

# Set the model to evaluation mode and disable gradient calculation for inference
model.eval()
torch.set_grad_enabled(False)

# Move model to appropriate device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())




# Time dataset initialization (loading from HDF5 into memory)
dataset_start = time.perf_counter()
dataset = HSCLegacyDataset(
    hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
    idx_list=list(range(95_000, 99_096)),
)
dataset_time = time.perf_counter() - dataset_start

# Time DataLoader creation (very fast, usually negligible)
loader_start = time.perf_counter()
train_loader = DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=4)
loader_time = time.perf_counter() - loader_start

# Time getting first batch (includes worker startup if num_workers > 0)
batch_start = time.perf_counter()
batch = next(iter(train_loader))
batch_time = time.perf_counter() - batch_start

# Get actual batch size (batch is a tuple: (hsc_images, legacy_images))
# Each element has shape (batch_size, channels, height, width)
actual_batch_size = batch[0].shape[0]  # or batch[1].shape[0]

total_time = dataset_time + loader_time + batch_time

print(f"Timing breakdown for {actual_batch_size} examples:")
print(f"  Dataset initialization (HDF5 → memory): {dataset_time:.4f} s")
print(f"  DataLoader creation:                    {loader_time:.4f} s")
print(f"  First batch retrieval:                  {batch_time:.4f} s")
print(f"  Total time:                             {total_time:.4f} s")


# Move images to device (they come from DataLoader on CPU by default)
hsc_images = batch[0].to(device)
legacy_images = batch[1].to(device)  # Fixed typo: was "legacy_iamges"

# Encode images with both encoders
with torch.no_grad():  # No gradients needed for inference
    hsc_embeddings_1 = model.encoder_1(hsc_images)
    legacy_embeddings_1 = model.encoder_1(legacy_images)
    hsc_embeddings_2 = model.encoder_2(hsc_images)
    legacy_embeddings_2 = model.encoder_2(legacy_images)

print(f"\nEncoding results:")
print(f"  HSC images shape:        {hsc_images.shape}")
print(f"  HSC embeddings 1 shape: {hsc_embeddings_1.shape}")
print(f"  HSC embeddings 2 shape: {hsc_embeddings_2.shape}")
print(f"  Legacy images shape:    {legacy_images.shape}")
print(f"  Legacy embeddings 1 shape: {legacy_embeddings_1.shape}")
print(f"  Legacy embeddings 2 shape: {legacy_embeddings_2.shape}")
print(f"\n  Embedding shape breakdown: (batch_size, seq_len, embed_dim)")
print(f"    - seq_len = spatial locations (H/32 * W/32 for ResNet18)")
print(f"    - embed_dim = cross_attention_dim = {model.hparams.cross_attention_dim}")

# Prepare embeddings for encoder 1
all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)
all_embeddings_1 = all_embeddings_1.flatten(start_dim=1)
print(f"\nEncoder 1 flattened embeddings shape: {all_embeddings_1.shape}")

# Prepare embeddings for encoder 2
all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)
all_embeddings_2 = all_embeddings_2.flatten(start_dim=1)
print(f"Encoder 2 flattened embeddings shape: {all_embeddings_2.shape}")


num_hsc = hsc_embeddings_1.shape[0]

hsc_flat_1 = all_embeddings_1[:num_hsc].cpu().numpy()
legacy_flat_1 = all_embeddings_1[num_hsc:].cpu().numpy()

hsc_flat_2 = all_embeddings_2[:num_hsc].cpu().numpy()
legacy_flat_2 = all_embeddings_2[num_hsc:].cpu().numpy()

### Latent prediction study

# Given a set of pairs hsc_image, legacy_image
# Calculate the galaxy/physics latents (looking at encoder_1)
# So we have z_hsc_ph, z_legacy_ph
# And calculate the instrument latents (looking at encoder_2)
# So we have z_hsc_ins, z_legacy_ins
# Then, train MLPs to predict:
#   1. z_hsc_ph -> z_legacy_ph (physics latents)
#   2. z_hsc_ins -> z_legacy_ins (instrument latents)

class LatentCrossPredictor(nn.Module):
    """
    Simple MLP to predict Legacy latents from HSC latents.
    Used for both physics and instrument latent prediction.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256, 128], dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Convert to tensors
z_hsc_ph = torch.from_numpy(hsc_flat_1).float()
z_legacy_ph = torch.from_numpy(legacy_flat_1).float()
z_hsc_ins = torch.from_numpy(hsc_flat_2).float()
z_legacy_ins = torch.from_numpy(legacy_flat_2).float()

print(f"\nLatent shapes:")
print(f"  z_hsc_ph: {z_hsc_ph.shape}")
print(f"  z_legacy_ph: {z_legacy_ph.shape}")
print(f"  z_hsc_ins: {z_hsc_ins.shape}")
print(f"  z_legacy_ins: {z_legacy_ins.shape}")

# Prepare data for training
# Task 1: Predict z_legacy_ph from z_hsc_ph (physics latents)
X_ph = z_hsc_ph  # Input: HSC physics latents
y_ph = z_legacy_ph  # Target: Legacy physics latents

# Task 2: Predict z_legacy_ins from z_hsc_ins (instrument latents)
X_ins = z_hsc_ins  # Input: HSC instrument latents
y_ins = z_legacy_ins  # Target: Legacy instrument latents

print(f"\nTraining data shapes:")
print(f"  Physics: X_ph={X_ph.shape}, y_ph={y_ph.shape}")
print(f"  Instrument: X_ins={X_ins.shape}, y_ins={y_ins.shape}")

# Create train/val split (same split for both tasks to keep pairs aligned)
train_size = int(0.8 * len(X_ph))
val_size = len(X_ph) - train_size

# Physics latents
X_ph_train, X_ph_val = torch.split(X_ph, [train_size, val_size])
y_ph_train, y_ph_val = torch.split(y_ph, [train_size, val_size])

# Instrument latents
X_ins_train, X_ins_val = torch.split(X_ins, [train_size, val_size])
y_ins_train, y_ins_val = torch.split(y_ins, [train_size, val_size])

# Create data loaders
train_dataset_ph = TensorDataset(X_ph_train, y_ph_train)
val_dataset_ph = TensorDataset(X_ph_val, y_ph_val)
train_dataset_ins = TensorDataset(X_ins_train, y_ins_train)
val_dataset_ins = TensorDataset(X_ins_val, y_ins_val)

train_loader_ph = DataLoader(train_dataset_ph, batch_size=256, shuffle=True)
val_loader_ph = DataLoader(val_dataset_ph, batch_size=256, shuffle=False)
train_loader_ins = DataLoader(train_dataset_ins, batch_size=256, shuffle=True)
val_loader_ins = DataLoader(val_dataset_ins, batch_size=256, shuffle=False)

# Initialize MLP models
input_dim_ph = X_ph.shape[1]
output_dim_ph = y_ph.shape[1]
mlp_physics = LatentCrossPredictor(
    input_dim=input_dim_ph,
    output_dim=output_dim_ph,
    hidden_dims=[512, 256, 128],
    dropout=0.1
).to(device)

input_dim_ins = X_ins.shape[1]
output_dim_ins = y_ins.shape[1]
mlp_instrument = LatentCrossPredictor(
    input_dim=input_dim_ins,
    output_dim=output_dim_ins,
    hidden_dims=[512, 256, 128],
    dropout=0.1
).to(device)

print(f"\nMLP Models:")
print(f"  Physics MLP: {input_dim_ph} -> {output_dim_ph}, Params: {sum(p.numel() for p in mlp_physics.parameters()):,}")
print(f"  Instrument MLP: {input_dim_ins} -> {output_dim_ins}, Params: {sum(p.numel() for p in mlp_instrument.parameters()):,}")

# Training setup
optimizer_ph = torch.optim.Adam(mlp_physics.parameters(), lr=1e-3)
optimizer_ins = torch.optim.Adam(mlp_instrument.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training function
def train_mlp(model, train_loader, val_loader, optimizer, num_epochs, model_name):
    best_val_loss = float('inf')

    print(f"\n{'='*60}")
    print(f"Training {model_name} MLP for {num_epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'latent_cross_predictor_{model_name}_best.pt')

    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    return best_val_loss

# Re-enable gradients for training the MLP models
torch.set_grad_enabled(True)

# Train both models
num_epochs = 50
best_val_loss_ph = train_mlp(mlp_physics, train_loader_ph, val_loader_ph, optimizer_ph, num_epochs, "physics")
best_val_loss_ins = train_mlp(mlp_instrument, train_loader_ins, val_loader_ins, optimizer_ins, num_epochs, "instrument")

# Load best models and evaluate
mlp_physics.load_state_dict(torch.load('latent_cross_predictor_physics_best.pt'))
mlp_instrument.load_state_dict(torch.load('latent_cross_predictor_instrument_best.pt'))
mlp_physics.eval()
mlp_instrument.eval()

# Evaluate physics model
print(f"\n{'='*60}")
print("Physics Latent Prediction Evaluation")
print(f"{'='*60}")
with torch.no_grad():
    X_ph_val_device = X_ph_val.to(device)
    y_ph_pred_val = mlp_physics(X_ph_val_device)
    mse_ph_val = criterion(y_ph_pred_val, y_ph_val.to(device)).item()

    # Compute per-dimension statistics
    mse_per_dim_ph = ((y_ph_pred_val - y_ph_val.to(device)) ** 2).mean(dim=0).cpu().numpy()

print(f"Final validation MSE: {mse_ph_val:.6f}")
print(f"  Mean MSE per dimension: {mse_per_dim_ph.mean():.6f}")
print(f"  Std MSE per dimension: {mse_per_dim_ph.std():.6f}")
print(f"  Min MSE per dimension: {mse_per_dim_ph.min():.6f}")
print(f"  Max MSE per dimension: {mse_per_dim_ph.max():.6f}")

# Evaluate instrument model
print(f"\n{'='*60}")
print("Instrument Latent Prediction Evaluation")
print(f"{'='*60}")
with torch.no_grad():
    X_ins_val_device = X_ins_val.to(device)
    y_ins_pred_val = mlp_instrument(X_ins_val_device)
    mse_ins_val = criterion(y_ins_pred_val, y_ins_val.to(device)).item()

    # Compute per-dimension statistics
    mse_per_dim_ins = ((y_ins_pred_val - y_ins_val.to(device)) ** 2).mean(dim=0).cpu().numpy()

print(f"Final validation MSE: {mse_ins_val:.6f}")
print(f"  Mean MSE per dimension: {mse_per_dim_ins.mean():.6f}")
print(f"  Std MSE per dimension: {mse_per_dim_ins.std():.6f}")
print(f"  Min MSE per dimension: {mse_per_dim_ins.min():.6f}")
print(f"  Max MSE per dimension: {mse_per_dim_ins.max():.6f}")

# Example predictions
print(f"\n{'='*60}")
print("Example Predictions (first 3 samples)")
print(f"{'='*60}")
with torch.no_grad():
    X_ph_sample = X_ph_val[:3].to(device)
    y_ph_true_sample = y_ph_val[:3].to(device)
    y_ph_pred_sample = mlp_physics(X_ph_sample)

    X_ins_sample = X_ins_val[:3].to(device)
    y_ins_true_sample = y_ins_val[:3].to(device)
    y_ins_pred_sample = mlp_instrument(X_ins_sample)

    print("\nPhysics latents:")
    for i in range(3):
        mse_sample = ((y_ph_pred_sample[i] - y_ph_true_sample[i]) ** 2).mean().item()
        print(f"  Sample {i+1}: MSE = {mse_sample:.6f}")

    print("\nInstrument latents:")
    for i in range(3):
        mse_sample = ((y_ins_pred_sample[i] - y_ins_true_sample[i]) ** 2).mean().item()
        print(f"  Sample {i+1}: MSE = {mse_sample:.6f}")
