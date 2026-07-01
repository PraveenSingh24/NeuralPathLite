import os
import re
import gc
import copy
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from diffusers import DDPMScheduler, UNet2DModel
from torch.optim.lr_scheduler import LambdaLR
import math

# ------------------ Configuration ------------------
class Config:
    image_size = 128
    batch_size = 16
    lr = 2e-4
    num_epochs = 300
    num_timesteps = 100
    film_channels = 32
    ema_decay = 0.999
    grad_clip = 1.0
    warmup_steps = 100

config = Config()

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, base_lr):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
    return LambdaLR(optimizer, lr_lambda)

# ------------------ Enhanced Dataset ------------------
class MazeDataset(Dataset):
    def __init__(self, maze_folder, path_folder):
        self.maze_paths = self._sorted_files(maze_folder)
        self.path_paths = self._sorted_files(path_folder)
        
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.maze_paths)  # Now explicitly defined

    def __getitem__(self, idx):
        maze = self._load_image(self.maze_paths[idx], 'RGB')
        path = self._load_image(self.path_paths[idx], 'L')
        
        return {
            'maze': maze,
            'path': path
        }

    def _sorted_files(self, folder):
        return sorted([os.path.join(folder, f) 
                      for f in os.listdir(folder) if f.endswith(('.png', '.txt'))])

    def _load_image(self, path, mode):
        return self.transform(Image.open(path).convert(mode))


# ------------------ Enhanced ResNet Encoder ------------------
class SpatialSoftmax(nn.Module):
    def forward(self, feature):
        B, C, H, W = feature.shape
        x_range = torch.linspace(-1.0, 1.0, W, device=feature.device)
        y_range = torch.linspace(-1.0, 1.0, H, device=feature.device)
        pos_x, pos_y = torch.meshgrid(x_range, y_range, indexing="xy")
        pos_x = pos_x.reshape(H * W)
        pos_y = pos_y.reshape(H * W)
        feature = feature.view(B, C, H * W)
        softmax_attention = torch.nn.functional.softmax(feature, dim=-1)
        expected_x = torch.sum(pos_x * softmax_attention, dim=-1)
        expected_y = torch.sum(pos_y * softmax_attention, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)

def replace_bn_with_gn(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups=32, num_channels=module.num_features)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module)
# ------------------ Corrected ResNetVisualEncoder ------------------
class ResNetVisualEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # Apply GroupNorm replacement
        replace_bn_with_gn(resnet)
        
        self.resnet = resnet
        self.spatial_softmax = SpatialSoftmax()
        self.fc = nn.Sequential(
            nn.Linear(512*2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        # (1) Process through ResNet
        features = self.resnet(x)  # [B, 512, H, W]
        
        # (2) Apply SpatialSoftmax 
        spatial_features = self.spatial_softmax(features)  # [B, 1024]
        
        # (3) Final projection
        return self.fc(spatial_features)  # [B, output_dim]

# ------------------ Verification ------------------
#encoder = ResNetVisualEncoder(output_dim=512)
#print(encoder)


# ------------------ Adaptive FiLM Layer ------------------
class FiLMGenerator(nn.Module):
    def __init__(self, cond_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim*2),
            nn.ReLU(),
            nn.Linear(cond_dim*2, out_dim*2)
        )
        
    def forward(self, x):
        params = self.mlp(x)
        gamma, beta = params.chunk(2, -1)
        return gamma.unsqueeze(-1).unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)

# ------------------ Modified UNet ------------------
class PathUNet(UNet2DModel):
    def __init__(self):
        super().__init__(
            in_channels= 3 + config.film_channels,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            norm_num_groups=32,
            attention_head_dim=8
        )

# ------------------ Training Utilities ------------------
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                s_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

def warmup_lr(step, warmup_steps, lr):
    return min(step / warmup_steps, 1.0) * lr


def load_checkpoint(model, encoder, film_gen, optimizer, ema, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    encoder.load_state_dict(checkpoint['encoder'])
    film_gen.load_state_dict(checkpoint['film'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    ema.shadow.load_state_dict(checkpoint['ema'])
    print(f"✅ Loaded checkpoint from {path}")
# ------------------ Training Loop ------------------
def train():
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model components
    model = PathUNet().to(device)
    ema = EMA(model, config.ema_decay)
    encoder = ResNetVisualEncoder(output_dim=128).to(device)
    film_gen = FiLMGenerator(128, config.film_channels).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()) + list(film_gen.parameters()),
        lr=config.lr,
        weight_decay=0.01
    )

    #total_steps = config.num_epochs * len(loader)
    #scheduler_lr = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps, config.lr)
    # Resume support
    #start_epoch = 0
    #if resume_from:
    #    load_checkpoint(model, encoder, film_gen, optimizer, ema, resume_from)
    #    start_epoch = int(resume_from.split("_")[-1].split(".")[0])
    # Scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=config.num_timesteps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    
    # Dataset
    dataset = MazeDataset("../ddpm/output_sets/maze", "../ddpm/output_sets/path")
    #dataset = MazeDataset("../mmNet/testData/multi_color/maze_with_sgd", "../mmNet/testData/multi_color/paths_only")
    #dataset = MazeDataset("../ddpm/mall/maze_with_sgd", "../ddpm/mall/paths_only")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    total_steps = config.num_epochs * len(loader)
    scheduler_lr = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps, config.lr)
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        total_mse = 0
     
        
        for batch_idx, batch in enumerate(loader):
            # Prepare data
            maze = batch['maze'].to(device)
            path = batch['path'].to(device)
            # Learning rate warmup
            #global_step = epoch * len(loader) + batch_idx
            #lr = warmup_lr(global_step, config.warmup_steps, config.lr)
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = lr
            scheduler_lr.step()
                
            # Conditioning
            maze_features = encoder(maze)
            
            # Add noise
            noise = torch.randn_like(path)
            timesteps = torch.randint(0, config.num_timesteps, (path.size(0),), device=device)
            noisy_path = scheduler.add_noise(path, noise, timesteps)
            
            # FiLM modulation
            gamma, beta = film_gen(maze_features)
            modulated = gamma * noisy_path.repeat(1, config.film_channels, 1, 1) + beta
            
            # Combine inputs
            model_input = torch.cat([modulated, maze], dim=1)
            
            # Predict noise
            noise_pred = model(model_input, timesteps).sample
            # Compute individual losses
            mse = F.mse_loss(noise_pred, noise)
            
            # Combine them into the total loss
            loss = mse 
            
            # Optimize
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema.update(model)
            
            total_mse += mse.item()
            
            # Free memory
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Epoch summary
        avg_mse = total_mse / len(loader)
        
        #print(f"Epoch {epoch+1}/{config.num_epochs} | Loss: {avg_loss:.5f} | LR: {lr:.2e}")
        print(f"Epoch {epoch+1}/{config.num_epochs} | Total Loss: {avg_mse:.5f}")

        
        # Save checkpoint
        if (epoch+1) % 5 == 0:
            torch.save({
                'model': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'encoder': encoder.state_dict(),
                'film': film_gen.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f"radhaRaman_bol_{epoch+1}.pth")

if __name__ == "__main__":
    #train("checkpoint_radha_45.pth")
    train()
