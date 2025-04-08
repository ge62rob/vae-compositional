# =========================================
# VAE for Tangram Images in Google Colab
# =========================================

# Install dependencies if needed (Colab usually has these installed)
# !pip install torch torchvision matplotlib

# TODO: check the source - original github code thats in the paper., train it with 2 curriculums, generate new images.

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define the VAE Architecture
# -------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # Encoder: input (1, 128, 128)
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)   # -> (32, 64, 64)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # -> (64, 32, 32)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # -> (128, 16, 16)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # -> (256, 8, 8)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder: maps latent vector back to image
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # -> (128, 16, 16)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # -> (64, 32, 32)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # -> (32, 64, 64)
        self.dec_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)     # -> (1, 128, 128)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = self.flatten(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(-1, 256, 8, 8)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        x_recon = torch.sigmoid(self.dec_conv4(h))
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# -------------------------------
# 2. Loss Function for the VAE
# -------------------------------
def vae_loss(x_recon, x, mu, logvar):
    # Reconstruction loss (BCE or MSE). Here, we use BCE since images are in [0,1].
    # For grayscale images, flatten the image.
    BCE = F.binary_cross_entropy(x_recon.view(x.size(0), -1),
                                 x.view(x.size(0), -1), reduction='sum')
    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (BCE + KLD) / x.size(0)

#############
#TODO:
#sample randomly and give it to the decoder to create synthetically
#How we are gonna use it
#train two different VAE's with 2 different curriculums
#qualitative inspection - maybe quantitative
#Evaluation on creating new data - generation not construction
#give top half of the image and ask it to complete bottom half
#FID score to evaluate
# post on slack

# search on github for generation function

###############



# -------------------------------
# 3. Training Function
# -------------------------------
def train_vae(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_batches = 0
    for imgs, _ in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    avg_loss = total_loss / total_batches
    return avg_loss

# -------------------------------
# 4. Inference & Visualization
# -------------------------------
def reconstruct_images_vae(model, images, device):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        recon, _, _ = model(images)
    return images.cpu(), recon.cpu()

def plot_vae_results(original, reconstructed, n=4):
    n = min(n, original.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(8, 3*n))
    for i in range(n):
        # Original Image
        axes[i, 0].imshow(original[i, 0].numpy(), cmap='gray')
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        # Reconstructed Image
        axes[i, 1].imshow(reconstructed[i, 0].numpy(), cmap='gray')
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()

# -------------------------------
# 5. Main Script for Colab
# -------------------------------

# Set parameters
data_path = "/content/dataset"    # Folder containing tangram subfolders
img_size = 128                    # Resize images to 128x128
batch_size = 32
num_epochs = 100
learning_rate = 1e-3 #1e-3 # 5e-4 denedim daha iyiydi / # 1e-5 di ilk orjinali
latent_dim = 32                  # Dimension of the latent space

# Create transforms & dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),  # values in [0,1]
])

dataset = datasets.ImageFolder(root=data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Create VAE model and optimizer
model = VAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
print("Starting VAE training...")
for epoch in range(num_epochs):
    avg_loss = train_vae(model, dataloader, optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the trained model
model_save_path = "/content/vae_tangram.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Inference & Visualization
# Get a batch from the dataloader
sample_iter = iter(dataloader)
imgs, _ = next(sample_iter)
original, reconstructed = reconstruct_images_vae(model, imgs, device)
plot_vae_results(original, reconstructed, n=4)
