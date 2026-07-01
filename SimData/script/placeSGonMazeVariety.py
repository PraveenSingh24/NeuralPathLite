"""
placeSGonMazeVariety.py
───────────────────────
Reads all maze images from maze_variety/ (nested type/variant folders),
places green Start (S) and blue Goal (G) pixel-art markers in free space
with VARYING distances, and saves the results.

Distance buckets:
  • close   : 15–35 pixels apart
  • medium  : 36–65 pixels apart
  • far     : 66–110 pixels apart

Output mirrors input structure under a new root:
    MazeVarietySG/
    ├── type1_mixed_bw/
    │   ├── density_low/
    │   │   ├── close/
    │   │   ├── medium/
    │   │   └── far/
    │   ...
"""

import os
import random
import math
from glob import glob
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────────────────────
# Pixel-art S and G patterns (5×5)
# ──────────────────────────────────────────────────────────────────────────────
PIXEL_S = [
    [1,1,1,1,1],
    [1,0,0,0,0],
    [1,1,1,1,1],
    [0,0,0,0,1],
    [1,1,1,1,1],
]
PIXEL_G = [
    [1,1,1,1,1],
    [1,0,0,0,0],
    [1,0,1,1,1],
    [1,0,0,0,1],
    [1,1,1,1,1],
]

SCALE       = 3
S_COLOR     = (0, 255, 0)    # green
G_COLOR     = (0, 0, 255)    # blue
LETTER_SIZE = 5 * SCALE      # 15 px

# Distance buckets (pixel distance between S and G centres)
DISTANCE_BUCKETS = {
    "close":  (15, 35),
    "medium": (36, 65),
    "far":    (66, 110),
}

SAMPLES_PER_BUCKET = 5   # per maze image per distance bucket

# ──────────────────────────────────────────────────────────────────────────────
# Drawing & free-space helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_pixel_letter(draw, pattern, cx, cy, color, scale):
    for dy, row in enumerate(pattern):
        for dx, val in enumerate(row):
            if val:
                x1 = cx + dx * scale
                y1 = cy + dy * scale
                x2 = x1 + scale - 1
                y2 = y1 + scale - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)


def is_free_pixel(pixel):
    """True if pixel is near-white (free space)."""
    return pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240


def is_region_free(image, x, y, w, h):
    """Check bounding box for free space."""
    img_w, img_h = image.size
    if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
        return False
    for dy in range(h):
        for dx in range(w):
            if not is_free_pixel(image.getpixel((x + dx, y + dy))):
                return False
    return True


def pixel_dist(sx, sy, gx, gy):
    """Euclidean distance between two points."""
    return math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)

# ──────────────────────────────────────────────────────────────────────────────
# Core placement
# ──────────────────────────────────────────────────────────────────────────────

def place_sg_on_image(base_image, dist_min, dist_max, max_attempts=2000):
    """
    Try to place S and G on free space such that the distance between
    their centres falls within [dist_min, dist_max].
    Returns (img_with_sg, success).
    """
    width, height = base_image.size
    margin = 5
    max_coord = width - LETTER_SIZE - margin

    if max_coord <= margin:
        return None, False

    for _ in range(max_attempts):
        sx = random.randint(margin, max_coord)
        sy = random.randint(margin, max_coord)
        gx = random.randint(margin, max_coord)
        gy = random.randint(margin, max_coord)

        d = pixel_dist(sx, sy, gx, gy)
        if d < dist_min or d > dist_max:
            continue

        if not is_region_free(base_image, sx, sy, LETTER_SIZE, LETTER_SIZE):
            continue
        if not is_region_free(base_image, gx, gy, LETTER_SIZE, LETTER_SIZE):
            continue

        # No overlap between S and G
        if abs(sx - gx) < LETTER_SIZE and abs(sy - gy) < LETTER_SIZE:
            continue

        # Draw
        img = base_image.copy()
        draw = ImageDraw.Draw(img)
        draw_pixel_letter(draw, PIXEL_S, sx, sy, S_COLOR, SCALE)
        draw_pixel_letter(draw, PIXEL_G, gx, gy, G_COLOR, SCALE)
        return img, True

    return None, False

# ──────────────────────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────────────────────

def process_all(input_root, output_root, samples_per_bucket=SAMPLES_PER_BUCKET):
    """
    Walk maze_variety/* structure, for each maze generate samples at
    each distance bucket, save under output_root with matching structure.
    """
    # Discover the type/variant folder tree
    type_dirs = sorted([
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    ])

    total_saved = 0

    for type_dir in type_dirs:
        type_path = os.path.join(input_root, type_dir)
        variant_dirs = sorted([
            v for v in os.listdir(type_path)
            if os.path.isdir(os.path.join(type_path, v))
        ])

        for var_dir in variant_dirs:
            var_path = os.path.join(type_path, var_dir)
            maze_files = sorted(glob(os.path.join(var_path, "*.png")))

            for bucket_name, (d_min, d_max) in DISTANCE_BUCKETS.items():
                out_dir = os.path.join(output_root, type_dir, var_dir, bucket_name)
                os.makedirs(out_dir, exist_ok=True)
                saved = 0

                for maze_path in maze_files:
                    base_image = Image.open(maze_path).convert("RGB")
                    maze_name = os.path.splitext(os.path.basename(maze_path))[0]

                    for s in range(1, samples_per_bucket + 1):
                        result, ok = place_sg_on_image(base_image, d_min, d_max)
                        if ok:
                            fname = f"{maze_name}_v{s}.png"
                            result.save(os.path.join(out_dir, fname))
                            saved += 1

                print(f"✅ {type_dir}/{var_dir}/{bucket_name} — {saved} images")
                total_saved += saved

    print(f"\n🎉 Total: {total_saved} images saved to {output_root}")


if __name__ == "__main__":
    INPUT_ROOT  = "/home/praveen_k/mmNet/testData/maze_variety/"
    OUTPUT_ROOT = "/home/praveen_k/mmNet/MazeVarietySG/"

    process_all(INPUT_ROOT, OUTPUT_ROOT, samples_per_bucket=5)
