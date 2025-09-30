import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# IDEA 1: Learned Frequency Dropout Module

# You can create a learnable mask or a small MLP to determine which frequency channels to suppress. During training, backpropagation will learn what to drop.
def add_high_frequency_noise(dct_batch, block_size=8, noise_std=0.1, target_band='high'):
    """
    Adds Gaussian noise to high-frequency DCT components in a batched DCT tensor.
    
    Args:
        dct_batch: torch.Tensor of shape [B, C, H, W]
        block_size: DCT block size (default: 8x8)
        noise_std: standard deviation of added noise
        target_band: 'high', 'mid', or 'low' (which frequency band to target)
    
    Returns:
        dct_batch_noisy: noisy version of input tensor
    """
    B, C, H, W = dct_batch.shape
    noisy = dct_batch.clone()
    
    # Loop over each 8x8 block
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            for u in range(block_size):
                for v in range(block_size):
                    freq_level = u + v
                    if target_band == 'high' and freq_level >= 10:
                        noise = torch.randn(B, C) * noise_std
                        noisy[:, :, i + u, j + v] += noise.to(dct_batch.device)
                    elif target_band == 'mid' and freq_level > 5:
                        noise = torch.randn(B, C) * noise_std
                        noisy[:, :, i + u, j + v] += noise.to(dct_batch.device)
                    elif target_band == 'low' and freq_level <= 10:
                        noise = torch.randn(B, C) * noise_std
                        noisy[:, :, i + u, j + v] += noise.to(dct_batch.device)
    return noisy


class LearnableFrequencyMask(nn.Module):
    def __init__(self, channels, height, width, init_scale=0.5):
        super().__init__()
        # Learnable mask shared across batch but applied to all images
        self.mask = nn.Parameter(init_scale * torch.rand(1, channels, height, width))  # shape [1, C, H, W]

    def forward(self, x):
        # x: [B, C, H, W]
        return x * torch.sigmoid(self.mask)


# ------------------- Preprocessing Utils -------------------
def rgb_to_ycbcr(img):
    img = img.convert('YCbCr')
    return np.array(img).transpose(2, 0, 1) / 255.0

def apply_dct_to_patch(patch):
    return dct(dct(patch.T, norm='ortho').T, norm='ortho')

def apply_idct_to_patch(patch):
    return idct(idct(patch.T, norm='ortho').T, norm='ortho')

def blockwise_dct(img, block_size=8):
    C, H, W = img.shape
    dct_coeffs = np.zeros_like(img)
    for c in range(C):
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                patch = img[c, i:i+block_size, j:j+block_size]
                if patch.shape != (block_size, block_size):
                    continue
                dct_coeffs[c, i:i+block_size, j:j+block_size] = apply_dct_to_patch(patch)
    return dct_coeffs

def blockwise_idct(coeffs, block_size=8):
    C, H, W = coeffs.shape
    img = np.zeros_like(coeffs)
    for c in range(C):
        for i in range(0, H, block_size):
            for j in range(0, W, block_size):
                patch = coeffs[c, i:i+block_size, j:j+block_size]
                if patch.shape != (block_size, block_size):
                    continue
                img[c, i:i+block_size, j:j+block_size] = apply_idct_to_patch(patch)
    return img



def compute_radial_energy_profile(dct_channel):
    H, W = dct_channel.shape
    center = (0, 0)
    y, x = np.ogrid[:H, :W]
    radius = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)
    max_radius = int(radius.max()) + 1
    radial_sum = np.zeros(max_radius)
    radial_count = np.zeros(max_radius)
    log_energy = np.log1p(np.abs(dct_channel))
    for r in range(max_radius):
        mask = (radius == r)
        radial_sum[r] = log_energy[mask].sum()
        radial_count[r] = mask.sum()
    return radial_sum / (radial_count + 1e-8)

def compute_band_energy(dct_channel, block_size=8):
    H, W = dct_channel.shape
    total_energy = np.sum(np.abs(dct_channel))
    low_mask = np.fromfunction(lambda u, v: (u + v) < 5, (block_size, block_size))
    mid_mask = np.fromfunction(lambda u, v: (5 <= (u + v)) & ((u + v) < 10), (block_size, block_size))
    high_mask = np.fromfunction(lambda u, v: (u + v) >= 10, (block_size, block_size))
    low, mid, high = 0, 0, 0
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            patch = dct_channel[i:i+block_size, j:j+block_size]
            if patch.shape != (block_size, block_size): continue
            low += np.sum(np.abs(patch[low_mask]))
            mid += np.sum(np.abs(patch[mid_mask]))
            high += np.sum(np.abs(patch[high_mask]))
    return {
        'low': low / total_energy,
        'mid': mid / total_energy,
        'high': high / total_energy
    }
# ------------------- Dataset -------------------
class DCTImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        img_ycbcr = rgb_to_ycbcr(img)
        dct_img = blockwise_dct(img_ycbcr)
        return torch.tensor(dct_img, dtype=torch.float32)

# ------------------- Model -------------------
# class SimpleDCTReconstructor(nn.Module):
#     def __init__(self, in_ch=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
#             nn.Conv2d(32, in_ch, 3, padding=1)
#         )

#     def forward(self, x):
#         return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))

class SimpleDCTReconstructor(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True), ResidualBlock(32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True), ResidualBlock(64)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(inplace=True), ResidualBlock(128)
        )

        # Bottleneck
        self.bottleneck = ResidualBlock(128)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True), ResidualBlock(64)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True), ResidualBlock(32)
        )
        self.dec1 = nn.Conv2d(32, in_ch, 3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(b) + e2
        d2 = self.dec2(d3) + e1
        out = self.dec1(d2)
        return out

# ------------------- Training -------------------
def train_model(data_folder='saved_images', epochs=50, batch_size=16, lr=1e-3, save_dir='idea1_2_very_high'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DCTImageDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    sample_dct = dataset[0]  # shape [C, H, W]
    C, H, W = sample_dct.shape
    # model_path = '/scratch/psm12/water/tree-ring-watermark/idea1_2_mid_low/dct_model_final_epoch35.pt'

    model = SimpleDCTReconstructor(in_ch=3).to(device)
    # model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    freq_mask = LearnableFrequencyMask(channels=C, height=H, width=W).to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            noisy_batch = add_high_frequency_noise(batch, noise_std=0.2, target_band='high')
            masked_dct = freq_mask(noisy_batch)
            recon = model(masked_dct)
            loss = criterion(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {running_loss / len(dataloader):.4f}")

        # Save sample reconstructions
        if (epoch+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                recon_np = recon[0].cpu().numpy()
                img_recon = blockwise_idct(recon_np)
                img_recon = Image.fromarray((img_recon.transpose(1,2,0) * 255).astype(np.uint8), 'YCbCr').convert('RGB')
                img_recon.save(os.path.join(save_dir, f'recon_epoch{epoch+1}.png'))
                recon_np = batch[0].cpu().numpy()
                img_recon = blockwise_idct(recon_np)
                img_recon = Image.fromarray((img_recon.transpose(1,2,0) * 255).astype(np.uint8), 'YCbCr').convert('RGB')
                img_recon.save(os.path.join(save_dir, f'image.png'))
                torch.save(model.state_dict(), os.path.join(save_dir, f'dct_model_final_epoch{epoch+1}.pt'))

    torch.save(model.state_dict(), os.path.join(save_dir, 'dct_model_final.pt'))




@torch.no_grad()
def run_inference(test_dir='test_images', save_dir='test_dir', model_path='idea1_2/dct_model_final.pt'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get test image paths
    image_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.png')])

    # Load one sample to infer shape
    img_sample = Image.open(image_paths[0])
    dct_sample = blockwise_dct(rgb_to_ycbcr(img_sample))
    C, H, W = dct_sample.shape

    # Initialize model and mask
    model = SimpleDCTReconstructor(in_ch=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    freq_mask = LearnableFrequencyMask(channels=C, height=H, width=W).to(device)
    freq_mask.eval()  # You can optionally load trained mask state_dict if saved separately

    for img_path in image_paths:
        img = Image.open(img_path)
        dct_np = blockwise_dct(rgb_to_ycbcr(img))
        dct_tensor = torch.tensor(dct_np, dtype=torch.float32).unsqueeze(0).to(device)  # [1, C, H, W]
        

        # Optionally add noise here (if needed in inference)
        noisy_img = add_high_frequency_noise(dct_tensor, noise_std=0.2, target_band='high')
        # masked_dct = freq_mask(noisy_img)
        recon = model(noisy_img)[0].cpu().numpy()  # shape [C, H, W]

        

        recon_img = blockwise_idct(recon)


        recon_img = np.clip(recon_img, 0, 1)
        # Visualize Y, Cb, Cr channels BEFORE transpose and conversion
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        channel_names = ['Y (Luminance)', 'Cb (Blue Chroma)', 'Cr (Red Chroma)']
        for i in range(3):
            axs[i].imshow(recon_img[i], cmap='gray')
            axs[i].set_title(channel_names[i])
            axs[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(test_dir, f'recon_channels_{os.path.basename(img_path)}'))

        recon_img = (recon_img.transpose(1,2,0) * 255).astype(np.uint8)
        recon_img = Image.fromarray(recon_img, mode='YCbCr').convert('RGB')
        
        out_path = os.path.join(save_dir, f'recon_{os.path.basename(img_path)}')
        recon_img.save(out_path)
        print(f"Saved: {out_path}")

if __name__ == '__main__':
    # run_inference(test_dir='test_dir', model_path='idea1_2_high_mid/dct_model_final_epoch20.pt')
    train_model()

