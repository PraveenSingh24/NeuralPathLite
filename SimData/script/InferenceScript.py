import torch
from torchvision import transforms
from PIL import Image
import time
import os
import glob
import sys
import numpy as np
from diffusers import DDPMScheduler
sys.path.insert(0, "/home/praveen_k/mmNet")
from mmNet import ResNetVisualEncoder, FiLMGenerator, Config, PathUNet  # Assuming model code is in model.py
from diffusers import UNet2DModel


torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# ------------------ Config ------------------
class Config:
    image_size = 128
    batch_size = 16
    lr = 2e-4
    num_timesteps = 20
    film_channels = 32
    ema_decay = 0.999
    grad_clip = 1.0
    warmup_steps = 100

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet2DModel(
    in_channels= 3 + config.film_channels,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    norm_num_groups=32,
    attention_head_dim=8
).to(device)

# ------------------ Load Single Image ------------------
def load_image(path, size, mode='RGB'):
    img = Image.open(path).convert(mode)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # [1, C, H, W]

def load_maze(path, size, mode='RGB'):
    img = Image.open(path).convert(mode)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)  # [1, C, H, W]

# ------------------ Inference ------------------
@torch.no_grad()
def infer(maze_path, ckpt_path="checkpoint_rama_70.pth", output_path="generated_path_output"):
    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    # Load components
    unet = model
    encoder = ResNetVisualEncoder(output_dim=128).to(device)
    film_gen = FiLMGenerator(128, config.film_channels).to(device)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['ema'])
    encoder.load_state_dict(ckpt['encoder'])
    film_gen.load_state_dict(ckpt['film'])

    unet.eval()
    encoder.eval()
    film_gen.eval()

    # Load maze files
    maze_files = glob.glob(os.path.join(maze_path, "*.png"))
    print(f"Found {len(maze_files)} maze images in {maze_path}.")

    for maze_path in maze_files:
        maze1 = load_image(maze_path, config.image_size).to(device)
        maze = load_maze(maze_path, config.image_size).to(device)
        # FiLM Conditioning
        maze_features = encoder(maze1)
        gamma, beta = film_gen(maze_features)

        # DDPM Scheduler
        scheduler = DDPMScheduler(
            num_train_timesteps=config.num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
        scheduler.set_timesteps(config.num_timesteps, device=device)

        sample = torch.randn((1, 1, config.image_size, config.image_size), device=device)

        # Timing
        if torch.cuda.is_available():
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            for t in scheduler.timesteps:
                modulated = gamma * sample.repeat(1, config.film_channels, 1, 1) + beta
                model_input = torch.cat([modulated, maze], dim=1)
                noise_pred = unet(model_input, t).sample
                sample = scheduler.step(noise_pred, t, sample).prev_sample

        if torch.cuda.is_available():
            ender.record()
            torch.cuda.synchronize()
            elapsed_time_ms = starter.elapsed_time(ender)
            print(f"Inference time: {elapsed_time_ms / 1000:.3f} seconds")

        # Save Output
        output = sample.squeeze().cpu().clamp(0, 1).numpy() * 255
        output_img = Image.fromarray(output.astype(np.uint8))

        base_name = os.path.splitext(os.path.basename(maze_path))[0]
        output_filename = os.path.join(output_path, base_name.replace("maze", "gen") + ".png")

        output_img.save(output_filename)
        print(f"Saved: {output_filename}")

# ------------------ Run Inference ------------------
if __name__ == "__main__":
    #infer("../ddpm/pathddpm/testData/output_set/maze_with_sg/", "radhaRaman_45.pth", "../ddpm/testResults/data2/")
    #infer("/home/praveen_k/InData/", "/home/praveen_k/mmNet/radhaRaman_80.pth", "/home/praveen_k/ddpm/OutInf/")
    infer("/home/praveen_k/mmNet/script/circle_rect_val_dataset/", "/home/praveen_k/mmNet/radhaRamanMC_55.pth", "/home/praveen_k/mmNet/Pout2/")
    #infer("/home/praveen_k/ddpm/office02/", "/home/praveen_k/mmNet/radhaRamanMrpb_185.pth", "/home/praveen_k/ddpm/testResults/oFFice02/")
