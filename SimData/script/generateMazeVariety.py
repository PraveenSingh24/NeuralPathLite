"""
generateMazeVariety.py
──────────────────────
Generates 4 types of mazes with green dot (start) and blue dot (goal)
placed in free space at varying distances. Ready for inference directly.

Maze Types:
  Type 1 — Mixed BW      : black circles + horizontal black rectangles
  Type 2 — Colored        : colored circles + colored bars
  Type 3 — Scattered Dots : many small rectangles scattered
  Type 4 — Room / Walls   : large rectangles forming corridors

For each type × variant, generates samples with 3 distance buckets
(close / medium / far) between start and goal.

Output:
    <root>/type1_mixed_bw/density_low/close/maze_000_v1.png
    <root>/type1_mixed_bw/density_low/medium/...
    ...
"""

import os
import random
import math
import numpy as np
from PIL import Image

RESOLUTION = 128
SAMPLES_PER_BUCKET = 5   # per base maze per distance bucket
BASE_MAZES = 20          # base mazes per variant

# Start / Goal dot parameters
DOT_RADIUS = 3
START_COLOR = [0, 255, 0]    # green
GOAL_COLOR  = [0, 0, 255]    # blue

# Distance buckets
DISTANCE_BUCKETS = {
    "close":  (15, 35),
    "medium": (36, 65),
    "far":    (66, 110),
}

# ──────────────────────────────────────────────────────────────────────────────
# Primitive obstacle drawers
# ──────────────────────────────────────────────────────────────────────────────

def draw_rect(img_np, x, y, w, h, color):
    x2 = min(x + w, img_np.shape[1])
    y2 = min(y + h, img_np.shape[0])
    img_np[max(y, 0):y2, max(x, 0):x2] = color


def draw_circle_np(img_np, cx, cy, r, color):
    Y, X = np.ogrid[:img_np.shape[0], :img_np.shape[1]]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img_np[mask] = color


# ──────────────────────────────────────────────────────────────────────────────
# Free-space & dot placement
# ──────────────────────────────────────────────────────────────────────────────

def is_free_at(img_np, cx, cy, r):
    """Check that a circle of radius r at (cx,cy) is entirely on white pixels
    and within bounds."""
    res = img_np.shape[0]
    if cx - r < 0 or cy - r < 0 or cx + r >= res or cy + r >= res:
        return False
    Y, X = np.ogrid[:res, :res]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    pixels = img_np[mask]
    return np.all(pixels > 240)


def place_dots(img_np, dist_min, dist_max, dot_r=DOT_RADIUS, max_attempts=2000):
    """
    Place green start dot and blue goal dot on free space with
    distance between centres in [dist_min, dist_max].
    Returns (img_with_dots, success).
    """
    res = img_np.shape[0]
    margin = dot_r + 2

    for _ in range(max_attempts):
        sx = random.randint(margin, res - margin - 1)
        sy = random.randint(margin, res - margin - 1)
        gx = random.randint(margin, res - margin - 1)
        gy = random.randint(margin, res - margin - 1)

        d = math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        if d < dist_min or d > dist_max:
            continue

        if not is_free_at(img_np, sx, sy, dot_r):
            continue
        if not is_free_at(img_np, gx, gy, dot_r):
            continue

        # No overlap between dots
        if d < 2 * dot_r + 2:
            continue

        result = img_np.copy()
        draw_circle_np(result, sx, sy, dot_r, START_COLOR)
        draw_circle_np(result, gx, gy, dot_r, GOAL_COLOR)
        return result, True

    return None, False


# ──────────────────────────────────────────────────────────────────────────────
# Maze generators
# ──────────────────────────────────────────────────────────────────────────────

def gen_type1(res, n_rect, n_circ, rect_size_range, circ_size_range):
    img = np.ones((res, res, 3), dtype=np.uint8) * 255
    color = [0, 0, 0]
    for _ in range(n_rect):
        w = random.randint(*rect_size_range[0])
        h = random.randint(*rect_size_range[1])
        x = random.randint(0, res - w)
        y = random.randint(0, res - h)
        draw_rect(img, x, y, w, h, color)
    for _ in range(n_circ):
        r = random.randint(*circ_size_range)
        cx = random.randint(r, res - r)
        cy = random.randint(r, res - r)
        draw_circle_np(img, cx, cy, r, color)
    return img

TYPE1_VARIANTS = {
    "density_low":  dict(n_rect=4,  n_circ=1,  rect_size_range=((10, 20), (2, 8)),   circ_size_range=(5, 15)),
    "density_med":  dict(n_rect=8,  n_circ=2,  rect_size_range=((10, 25), (2, 10)),  circ_size_range=(5, 25)),
    "density_high": dict(n_rect=14, n_circ=4,  rect_size_range=((10, 30), (3, 12)),  circ_size_range=(8, 28)),
}

def gen_type2(res, n_rect, n_circ, rect_color, circ_color, rect_size_range, circ_size_range):
    img = np.ones((res, res, 3), dtype=np.uint8) * 255
    for _ in range(n_rect):
        w = random.randint(*rect_size_range[0])
        h = random.randint(*rect_size_range[1])
        x = random.randint(0, res - w)
        y = random.randint(0, res - h)
        draw_rect(img, x, y, w, h, rect_color)
    for _ in range(n_circ):
        r = random.randint(*circ_size_range)
        cx = random.randint(r, res - r)
        cy = random.randint(r, res - r)
        draw_circle_np(img, cx, cy, r, circ_color)
    return img

TYPE2_VARIANTS = {
    "pink_orange": dict(n_rect=4, n_circ=3, rect_color=[255,105,180], circ_color=[255,165,0],
                        rect_size_range=((10,25),(2,10)), circ_size_range=(5,25)),
    "purple_yellow": dict(n_rect=5, n_circ=3, rect_color=[128,0,128], circ_color=[255,215,0],
                          rect_size_range=((10,25),(2,10)), circ_size_range=(5,25)),
    "cyan_red": dict(n_rect=5, n_circ=3, rect_color=[0,200,200], circ_color=[220,20,60],
                     rect_size_range=((10,25),(2,10)), circ_size_range=(5,25)),
}

def gen_type3(res, n_dots, dot_size_range, color):
    img = np.ones((res, res, 3), dtype=np.uint8) * 255
    for _ in range(n_dots):
        w = random.randint(*dot_size_range[0])
        h = random.randint(*dot_size_range[1])
        x = random.randint(0, res - w)
        y = random.randint(0, res - h)
        draw_rect(img, x, y, w, h, color)
    return img

TYPE3_VARIANTS = {
    "density_low":  dict(n_dots=15, dot_size_range=((2,6),(2,6)),   color=[0,0,0]),
    "density_med":  dict(n_dots=30, dot_size_range=((2,8),(2,8)),   color=[0,0,0]),
    "density_high": dict(n_dots=50, dot_size_range=((3,10),(3,10)), color=[0,0,0]),
}

def gen_type4(res, n_walls, wall_size_range, color):
    img = np.ones((res, res, 3), dtype=np.uint8) * 255
    for _ in range(n_walls):
        if random.random() < 0.5:
            w = random.randint(*wall_size_range[0])
            h = random.randint(*wall_size_range[1])
        else:
            h = random.randint(*wall_size_range[0])
            w = random.randint(*wall_size_range[1])
        x = random.randint(0, max(0, res - w))
        y = random.randint(0, max(0, res - h))
        draw_rect(img, x, y, w, h, color)
    return img

TYPE4_VARIANTS = {
    "few_rooms":      dict(n_walls=3,  wall_size_range=((30,60),(5,15)), color=[0,0,0]),
    "moderate_rooms": dict(n_walls=6,  wall_size_range=((25,55),(5,15)), color=[0,0,0]),
    "dense_rooms":    dict(n_walls=10, wall_size_range=((20,50),(5,12)), color=[0,0,0]),
}


# ──────────────────────────────────────────────────────────────────────────────
# Generate & Save
# ──────────────────────────────────────────────────────────────────────────────

def generate_all(output_root, base_mazes=BASE_MAZES, samples_per_bucket=SAMPLES_PER_BUCKET):
    configs = [
        ("type1_mixed_bw",  gen_type1, TYPE1_VARIANTS),
        ("type2_colored",   gen_type2, TYPE2_VARIANTS),
        ("type3_scattered", gen_type3, TYPE3_VARIANTS),
        ("type4_rooms",     gen_type4, TYPE4_VARIANTS),
    ]

    total = 0
    for type_name, gen_fn, variants in configs:
        for var_name, params in variants.items():
            # Generate base mazes first
            base_images = []
            for _ in range(base_mazes):
                base_images.append(gen_fn(RESOLUTION, **params))

            # For each distance bucket, place dots on each base maze
            for bucket_name, (d_min, d_max) in DISTANCE_BUCKETS.items():
                folder = os.path.join(output_root, type_name, var_name, bucket_name)
                os.makedirs(folder, exist_ok=True)
                saved = 0

                for mi, base_img in enumerate(base_images):
                    for s in range(1, samples_per_bucket + 1):
                        result, ok = place_dots(base_img, d_min, d_max)
                        if ok:
                            fname = f"maze_{mi:03d}_v{s}.png"
                            Image.fromarray(result).save(os.path.join(folder, fname))
                            saved += 1

                print(f"✅ {type_name}/{var_name}/{bucket_name} — {saved} images")
                total += saved

    print(f"\n🎉 Total: {total} images saved to {output_root}")


if __name__ == "__main__":
    OUTPUT_ROOT = "/home/praveen_k/mmNet/MazeVarietySG/"
    generate_all(OUTPUT_ROOT, base_mazes=100, samples_per_bucket=1)
