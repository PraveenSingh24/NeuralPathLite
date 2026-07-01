import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import DDPMScheduler, UNet2DModel
import os
import glob
import sys
import time

# Add parent directory so we can import mmNet
sys.path.insert(0, "/home/praveen_k/mmNet")
from mmNet import ResNetVisualEncoder, FiLMGenerator

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────────────────────────────────────
# Config (must match training)
# ──────────────────────────────────────────────────────────────────────────────
class Config:
    image_size   = 128
    num_timesteps = 400
    film_channels = 32

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Model (built once, weights loaded once)
# ──────────────────────────────────────────────────────────────────────────────
unet = UNet2DModel(
    in_channels=3 + config.film_channels,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    norm_num_groups=32,
    attention_head_dim=8,
).to(device)

encoder  = ResNetVisualEncoder(output_dim=128).to(device)
film_gen = FiLMGenerator(128, config.film_channels).to(device)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
])

def load_image(path):
    return _transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def load_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['ema'])
    encoder.load_state_dict(ckpt['encoder'])
    film_gen.load_state_dict(ckpt['film'])
    unet.eval(); encoder.eval(); film_gen.eval()
    print(f"✅ Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def infer_folder(input_dir, output_dir):
    """Run inference on every *.png in `input_dir`, save results to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    maze_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not maze_files:
        print(f"  ⚠️  No images in {input_dir}")
        return 0

    count = 0
    for maze_path in maze_files:
        maze = load_image(maze_path)

        # FiLM conditioning
        maze_features = encoder(maze)
        gamma, beta = film_gen(maze_features)

        # DDPM reverse process
        scheduler = DDPMScheduler(
            num_train_timesteps=config.num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        scheduler.set_timesteps(config.num_timesteps, device=device)
        sample = torch.randn((1, 1, config.image_size, config.image_size), device=device)

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            for t in scheduler.timesteps:
                modulated = gamma * sample.repeat(1, config.film_channels, 1, 1) + beta
                model_input = torch.cat([modulated, maze], dim=1)
                noise_pred = unet(model_input, t).sample
                sample = scheduler.step(noise_pred, t, sample).prev_sample

        # Save
        output = sample.squeeze().cpu().clamp(0, 1).numpy() * 255
        output_img = Image.fromarray(output.astype(np.uint8))
        base_name = os.path.splitext(os.path.basename(maze_path))[0]
        output_img.save(os.path.join(output_dir, base_name + ".png"))
        count += 1

    return count

# ──────────────────────────────────────────────────────────────────────────────
# Batch: generic recursive walk — works for ANY folder depth
# ──────────────────────────────────────────────────────────────────────────────

def batch_inference(maze_root, output_root, ckpt_path):
    """
    Recursively walk `maze_root`. For every leaf folder that contains
    *.png files, run inference and mirror the relative path under
    `output_root`.  Works for flat, 2-level, 3-level, or deeper trees.
    """
    load_weights(ckpt_path)

    total_images = 0
    total_folders = 0
    t0 = time.time()

    for dirpath, dirnames, filenames in os.walk(maze_root):
        # Skip folders that don't contain any .png files
        png_files = [f for f in filenames if f.lower().endswith(".png")]
        if not png_files:
            continue

        # Compute the relative path and mirror it under output_root
        rel = os.path.relpath(dirpath, maze_root)
        out_dir = os.path.join(output_root, rel) if rel != "." else output_root

        print(f"▶ {rel}  ({len(png_files)} images)")
        n = infer_folder(dirpath, out_dir)
        print(f"   → {n} images saved to {out_dir}\n")

        total_images  += n
        total_folders += 1

    elapsed = time.time() - t0
    print(f"🎉 Done!  {total_folders} folders, {total_images} images, "
          f"{elapsed:.1f}s total ({elapsed/max(total_images,1):.2f}s/img)")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MAZE_ROOT   = "/home/praveen_k/mmNet/MazeVarietySG/"
    OUTPUT_ROOT = "/home/praveen_k/mmNet/PathVarietySG/"
    CHECKPOINT  = "/home/praveen_k/mmNet/radhaRamanMC_75.pth"

    batch_inference(MAZE_ROOT, OUTPUT_ROOT, CHECKPOINT)

