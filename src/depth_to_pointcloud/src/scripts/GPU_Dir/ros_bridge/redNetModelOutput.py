#!/usr/bin/env python3
import os
import time
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import RedNet_model  # your RedNet_model.py from training repo


# ---------------- CONFIGURATION ----------------
SHARED_DIR = "/home/praveen_k/ros_bridge/shared/"

CKPT_PATH = "/home/praveen_k/RadhaGovind/model_bestRN1.pth"
NUM_CLASSES = 2
IMAGE_H = 480
IMAGE_W = 640

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------------------

HISTORY_DIR = os.path.join(SHARED_DIR, "history")
os.makedirs(HISTORY_DIR, exist_ok=True)


# ---------------- UTILS ----------------
def color_label(label_arr, num_classes=NUM_CLASSES):
    cmap = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
    ], dtype=np.uint8)

    if num_classes > cmap.shape[0]:
        rng = np.random.RandomState(0)
        extra = rng.randint(0, 255, size=(num_classes - cmap.shape[0], 3), dtype=np.uint8)
        cmap = np.vstack([cmap, extra])

    h, w = label_arr.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for c in range(num_classes):
        mask = (label_arr == c)
        color[mask] = cmap[c]

    return color


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format.")

    # remove module. if present
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_sd[new_key] = v

    model.load_state_dict(new_sd, strict=True)
    return model


def preprocess_rgb(img_pil):
    tf = transforms.Compose([
        transforms.Resize((IMAGE_H, IMAGE_W), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img_pil)


def preprocess_depth(depth_path):
    d_im = Image.open(depth_path)
    d_np = np.array(d_im)

    # Convert to float [0..1]
    if np.issubdtype(d_np.dtype, np.integer):
        if d_np.dtype == np.uint16:
            d_np = d_np.astype(np.float32) / 65535.0
        else:
            d_np = d_np.astype(np.float32) / 255.0
    else:
        d_np = d_np.astype(np.float32)
        vmax = d_np.max() if d_np.size else 1.0
        if vmax > 1.0:
            d_np = d_np / (vmax + 1e-8)

    # ensure single channel
    if d_np.ndim == 3:
        d_np = d_np[..., 0]

    t = torch.from_numpy(d_np).unsqueeze(0).float()  # [1,H,W]

    # resize
    t = F.interpolate(
        t.unsqueeze(0),
        size=(IMAGE_H, IMAGE_W),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    if t.shape[0] > 1:
        t = t.mean(dim=0, keepdim=True)

    return t


# ---------------- MAIN PROCESS ----------------
def main():
    print("🚀 RedNet segmentation daemon started")
    print(f"📂 Shared dir: {SHARED_DIR}")
    print(f"🧠 Model ckpt: {CKPT_PATH}")
    print(f"🖥️ Device: {DEVICE}")

    # Load model once
    model = RedNet_model.RedNet(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model = load_checkpoint(model, CKPT_PATH, DEVICE)
    model.eval()

    while True:
        task_file = os.path.join(SHARED_DIR, "seg_task.json")
        result_file_tmp = os.path.join(SHARED_DIR, "seg_result.json.tmp")
        result_file = os.path.join(SHARED_DIR, "seg_result.json")

        if os.path.exists(task_file):
            try:
                time.sleep(0.05)  # small wait for write completion

                # 1) Read task
                with open(task_file, "r") as f:
                    data = json.load(f)

                # 2) Delete task immediately (critical race-condition fix)
                os.remove(task_file)

                # 3) Extract task info
                task_id = data.get("task_id", str(int(time.time())))
                rgb_file = data.get("rgb_file", "rgb.png")
                depth_file = data.get("depth_file", "depth.png")

                rgb_path = os.path.join(SHARED_DIR, rgb_file)
                depth_path = os.path.join(SHARED_DIR, depth_file)

                print(f"📥 Processing Seg Task {task_id}")
                print(f"   RGB:   {rgb_path}")
                print(f"   Depth: {depth_path}")

                if not os.path.exists(rgb_path):
                    print(f"❌ Missing RGB file: {rgb_path}")
                    continue

                if not os.path.exists(depth_path):
                    print(f"❌ Missing depth file: {depth_path}")
                    continue

                # 4) Load + preprocess
                rgb_img = Image.open(rgb_path).convert("RGB")
                rgb_tensor = preprocess_rgb(rgb_img).unsqueeze(0).to(DEVICE)

                depth_tensor = preprocess_depth(depth_path).unsqueeze(0).to(DEVICE)

                # 5) Inference
                with torch.no_grad():
                    output = model(rgb_tensor, depth_tensor, False)
                    main_out = output[0] if isinstance(output, (list, tuple)) else output
                    main_up = F.interpolate(
                        main_out,
                        size=(IMAGE_H, IMAGE_W),
                        mode="bilinear",
                        align_corners=False
                    )
                    pred_mask = torch.argmax(main_up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                # 6) Save outputs
                mask_name = f"segmask_{task_id}.png"
                viz_name = f"segviz_{task_id}.png"

                mask_path = os.path.join(SHARED_DIR, mask_name)
                viz_path = os.path.join(SHARED_DIR, viz_name)

                Image.fromarray(pred_mask).save(mask_path)

                color_vis = color_label(pred_mask, num_classes=NUM_CLASSES)
                Image.fromarray(color_vis).save(viz_path)

                # also archive
                hist_mask = os.path.join(HISTORY_DIR, mask_name)
                hist_viz = os.path.join(HISTORY_DIR, viz_name)

                try:
                    Image.fromarray(pred_mask).save(hist_mask)
                    Image.fromarray(color_vis).save(hist_viz)
                except Exception as e:
                    print(f"⚠️ Could not save history copy: {e}")

                # 7) Write result.json atomically
                with open(result_file_tmp, "w") as f:
                    json.dump({
                        "task_id": task_id,
                        "mask_file": mask_name,
                        "viz_file": viz_name,
                        "num_classes": NUM_CLASSES
                    }, f)

                os.rename(result_file_tmp, result_file)
                print(f"📤 Sent seg result for Task {task_id}")

            except Exception as e:
                print(f"❌ Error: {e}")

                # cleanup
                if os.path.exists(task_file):
                    os.remove(task_file)

        time.sleep(0.05)


if __name__ == "__main__":
    main()
