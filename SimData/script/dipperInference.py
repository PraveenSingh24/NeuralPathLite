"""
dipperInference.py
──────────────────
Self-contained DiPPeR (Diffusion-based 2D Path Planner) inference script
for benchmarking against our maze dataset.

Architecture (from ICRA 2024 paper):
  - Vision encoder : ResNet18 (BatchNorm → GroupNorm)
  - Noise predictor: ConditionalUnet1D (1D temporal UNet with FiLM)
  - Scheduler      : DDPMScheduler (squaredcos_cap_v2)

Input:  maze image (resized to 100×100) + start (x,y) + goal (x,y)
Output: sequence of (x,y) waypoints  (pred_horizon steps)

This script:
  1. Detects green (start) and blue (goal) markers in maze images
  2. Runs DiPPeR inference for each image
  3. Measures and reports per-image and average inference time
  4. Saves path visualization overlaid on maze

Usage:
    python dipperInference.py

Requires:
    - Pretrained weights at CHECKPOINT_PATH (download from HuggingFace)
    - conda activate myenv  (torch, diffusers, torchvision)
"""

import os
import sys
import math
import time
import glob
import csv
import numpy as np
from typing import Tuple, Union, Callable

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ══════════════════════════════════════════════════════════════════════════════
# DiPPeR Model Architecture  (from official notebook)
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 diffusion_step_embed_dim=256,
                 down_dims=[256, 512, 1024],
                 kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out * 2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, global_cond=None):
        sample = sample.moveaxis(-1, -2)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = x.moveaxis(-1, -2)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# ResNet18 Vision Encoder  (BN → GN for EMA compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def get_resnet(name='resnet18', weights=None, **kwargs):
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = nn.Identity()
    return resnet


def replace_submodules(root_module, predicate, func):
    bn_list = [k.split('.') for k, m
               in root_module.named_modules(remove_duplicate=True)
               if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def replace_bn_with_gn(root_module, features_per_group=16):
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features // features_per_group,
            num_channels=x.num_features)
    )
    return root_module


# ══════════════════════════════════════════════════════════════════════════════
# Data normalisation  (DiPPeR uses [-5, 5] world coords → [-1, 1])
# ══════════════════════════════════════════════════════════════════════════════

STATS = {'min': -5., 'max': 5.}

def normalize_data(data, stats=STATS):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats=STATS):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


# ══════════════════════════════════════════════════════════════════════════════
# Start / Goal detection from maze image
# ══════════════════════════════════════════════════════════════════════════════

def detect_marker(img_np, target_color, tolerance=80):
    """
    Detect a coloured marker (start=green, goal=blue) in an RGB image.
    Returns (cx, cy) in pixel coords, or None if not found.
    """
    diff = np.abs(img_np.astype(np.int16) - np.array(target_color, dtype=np.int16))
    mask = np.all(diff < tolerance, axis=-1)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def pixel_to_world(px, py, img_size=100, world_range=(-5, 5)):
    """Convert pixel (px, py) to DiPPeR world coordinates."""
    wmin, wmax = world_range
    scale = (wmax - wmin) / img_size
    wx = wmin + px * scale
    wy = wmin + py * scale
    return wx, wy


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

DIPPER_IMAGE_SIZE    = 100      # DiPPeR uses 100×100 images
OBS_HORIZON          = 1
ACTION_DIM           = 2
VISION_FEATURE_DIM   = 512
START_DIM            = 2
GOAL_DIM             = 2
OBS_DIM              = VISION_FEATURE_DIM + START_DIM + GOAL_DIM
PRED_HORIZON         = 160
NUM_DIFFUSION_ITERS  = 100


# ══════════════════════════════════════════════════════════════════════════════
# Build model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(device):
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=ACTION_DIM,
        global_cond_dim=OBS_DIM * OBS_HORIZON
    )

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net,
    })
    nets = nets.to(device)
    return nets


def load_checkpoint(nets, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    nets.load_state_dict(state_dict)
    nets.eval()
    print(f"✅ Loaded DiPPeR checkpoint: {ckpt_path}")
    return nets


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_single(nets, maze_img_np, start_world, goal_world, device,
                 num_diffusion_iters=NUM_DIFFUSION_ITERS,
                 path_steps=PRED_HORIZON):
    """
    Run DiPPeR inference on a single maze.

    Args:
        nets:  DiPPeR model dict
        maze_img_np: (100, 100, 3) uint8 numpy array
        start_world: (x, y) in world coords [-5, 5]
        goal_world:  (x, y) in world coords [-5, 5]

    Returns:
        path_world: (N, 2) numpy array of waypoints in world coords
        elapsed_ms: inference time in milliseconds
    """
    # Prepare image: (1, 3, 100, 100)
    img = np.moveaxis(maze_img_np.astype(np.float32) / 255.0, -1, 0)
    images = np.stack([img for _ in range(OBS_HORIZON)])
    nimages = torch.from_numpy(images).to(device, dtype=torch.float32)

    # Prepare start/goal
    start_np = normalize_data(np.array(start_world, dtype=np.float32).reshape(1, 2))
    goal_np = normalize_data(np.array(goal_world, dtype=np.float32).reshape(1, 2))
    nstart = torch.from_numpy(start_np).to(device, dtype=torch.float32)
    ngoal = torch.from_numpy(goal_np).to(device, dtype=torch.float32)

    # Vision features
    image_features = nets['vision_encoder'](nimages)
    obs_features = torch.cat([image_features, nstart, ngoal], dim=-1)
    obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

    # Initialise noisy action
    noisy_action = torch.randn(
        (1, path_steps, ACTION_DIM), device=device)
    noisy_action[0, 0, :] = torch.tensor(start_np.flatten())
    noisy_action[0, -1, :] = torch.tensor(goal_np.flatten())
    naction = noisy_action

    # Scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    noise_scheduler.set_timesteps(num_diffusion_iters)

    # ── Timed inference ──────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

    t0 = time.perf_counter()

    for k in noise_scheduler.timesteps:
        naction[0, 0, :] = torch.tensor(start_np.flatten()).to(device)
        naction[0, -1, :] = torch.tensor(goal_np.flatten()).to(device)

        noise_pred = nets['noise_pred_net'](
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )

        naction[0, 0, :] = torch.tensor(start_np.flatten()).to(device)
        naction[0, -1, :] = torch.tensor(goal_np.flatten()).to(device)

        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample

        naction[0, 0, :] = torch.tensor(start_np.flatten()).to(device)
        naction[0, -1, :] = torch.tensor(goal_np.flatten()).to(device)

    t1 = time.perf_counter()

    if torch.cuda.is_available():
        ender.record()
        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender)
    else:
        elapsed_ms = (t1 - t0) * 1000.0

    # Unnormalise
    path_norm = naction.detach().cpu().numpy()[0]
    path_world = unnormalize_data(path_norm)

    return path_world, elapsed_ms


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation: draw path on maze
# ══════════════════════════════════════════════════════════════════════════════

def draw_path_on_maze(maze_img_np, path_world, img_size=100):
    """Overlay waypoint path (red dots) on maze image."""
    vis = maze_img_np.copy()
    origin = np.array([-5., -5.])
    scale = 0.1  # 1 pixel = 0.1m

    for wx, wy in path_world:
        px = int((wx - origin[0]) / scale)
        py = int((wy - origin[1]) / scale)
        if 0 <= px < img_size and 0 <= py < img_size:
            # Draw 3×3 red dot
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < img_size and 0 <= ny < img_size:
                        vis[ny, nx] = [255, 0, 0]
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# Batch inference with timing
# ══════════════════════════════════════════════════════════════════════════════

def batch_inference(input_root, output_root, ckpt_path,
                    num_diffusion_iters=NUM_DIFFUSION_ITERS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build & load model
    nets = build_model(device)
    nets = load_checkpoint(nets, ckpt_path, device)

    total_images = 0
    total_time_ms = 0.0
    timing_rows = []

    for dirpath, dirnames, filenames in os.walk(input_root):
        png_files = sorted([f for f in filenames if f.lower().endswith(".png")])
        if not png_files:
            continue

        rel = os.path.relpath(dirpath, input_root)
        out_dir = os.path.join(output_root, rel) if rel != "." else output_root
        os.makedirs(out_dir, exist_ok=True)

        print(f"▶ {rel}  ({len(png_files)} images)")

        for fname in png_files:
            fpath = os.path.join(dirpath, fname)

            # Load and resize to DiPPeR's expected 100×100
            pil_img = Image.open(fpath).convert("RGB")
            pil_100 = pil_img.resize((DIPPER_IMAGE_SIZE, DIPPER_IMAGE_SIZE),
                                     Image.LANCZOS)
            img_np = np.array(pil_100)

            # Detect start (green) and goal (blue)
            start_px = detect_marker(img_np, [0, 255, 0], tolerance=100)
            goal_px = detect_marker(img_np, [0, 0, 255], tolerance=100)

            if start_px is None or goal_px is None:
                print(f"  ⚠️  Skipping {fname}: markers not detected "
                      f"(S={start_px}, G={goal_px})")
                continue

            # Convert pixel → world coords
            start_world = pixel_to_world(start_px[0], start_px[1])
            goal_world = pixel_to_world(goal_px[0], goal_px[1])

            # Run inference
            path_world, elapsed_ms = infer_single(
                nets, img_np, start_world, goal_world, device,
                num_diffusion_iters=num_diffusion_iters
            )

            # Save visualisation
            vis = draw_path_on_maze(img_np, path_world)
            Image.fromarray(vis).save(os.path.join(out_dir, fname))

            total_images += 1
            total_time_ms += elapsed_ms

            timing_rows.append({
                "file": os.path.join(rel, fname),
                "inference_ms": f"{elapsed_ms:.2f}",
            })

        folder_count = sum(1 for r in timing_rows
                          if r["file"].startswith(rel + "/"))
        print(f"   → {folder_count} images processed\n")

    # ── Summary ──
    avg_ms = total_time_ms / max(total_images, 1)
    print(f"\n{'='*60}")
    print(f"DiPPeR Inference Benchmark")
    print(f"{'='*60}")
    print(f"  Total images     : {total_images}")
    print(f"  Total time       : {total_time_ms/1000:.2f} s")
    print(f"  Avg per image    : {avg_ms:.2f} ms ({avg_ms/1000:.3f} s)")
    print(f"  Diffusion steps  : {num_diffusion_iters}")
    print(f"{'='*60}")

    # Write timing CSV
    os.makedirs(output_root, exist_ok=True)
    csv_path = os.path.join(output_root, "dipper_timing.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "inference_ms"])
        writer.writeheader()
        writer.writerows(timing_rows)
        # Summary row
        writer.writerow({
            "file": "AVERAGE",
            "inference_ms": f"{avg_ms:.2f}",
        })
    print(f"📋 Timing CSV: {csv_path}")
    print(f"🎉 Done!")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Input: our maze dataset with S/G markers
    INPUT_ROOT   = "/home/praveen_k/mmNet/MazeWithArrows/"
    OUTPUT_ROOT  = "/home/praveen_k/mmNet/DiPPeR_PathOut/"
    CHECKPOINT   = "/home/praveen_k/mmNet/DiPPeR/dipper_pretrained.pth"

    # DiPPeR paper uses 100 diffusion steps for fast inference
    # (they report 0.4s per image at 100 steps)
    batch_inference(INPUT_ROOT, OUTPUT_ROOT, CHECKPOINT,
                    num_diffusion_iters=100)
