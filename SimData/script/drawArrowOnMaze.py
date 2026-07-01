"""
drawArrowOnMaze.py
──────────────────
Draws ARROW and SPECIAL-SYMBOL markers at Start and Goal positions on maze
images, keeping the same baseline green/blue colour scheme.

Generates an ablation-style dataset with 4 experiment groups:
  1. vary_symbol_style  — different arrow / symbol pairings
  2. vary_size          — pixel-art scale 2–5
  3. vary_s_color       — start marker colour variants
  4. vary_g_color       — goal  marker colour variants

Usage:
    python drawArrowOnMaze.py
"""

import os
import csv
import random
from glob import glob
from PIL import Image, ImageDraw

# ══════════════════════════════════════════════════════════════════════════════
# 7×7 Pixel-Art Bitmaps  —  Arrows
# ══════════════════════════════════════════════════════════════════════════════

ARROW_RIGHT = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

ARROW_DOWN = [
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 0, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
]

ARROW_UP = [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 0, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
]

ARROW_LEFT = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

# ══════════════════════════════════════════════════════════════════════════════
# 7×7 Pixel-Art Bitmaps  —  Special Symbols
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_STAR = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
]

SYMBOL_TRIANGLE = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

SYMBOL_DIAMOND = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
]

SYMBOL_CROSS = [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
]

SYMBOL_CIRCLE = [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0],
]

# ══════════════════════════════════════════════════════════════════════════════
# Registry: name → bitmap (for clean lookup in experiments)
# ══════════════════════════════════════════════════════════════════════════════

PATTERN_MAP = {
    "arrow_right":      ARROW_RIGHT,
    "arrow_down":       ARROW_DOWN,
    "arrow_up":         ARROW_UP,
    "arrow_left":       ARROW_LEFT,
    "star":             SYMBOL_STAR,
    "triangle":         SYMBOL_TRIANGLE,
    "diamond":          SYMBOL_DIAMOND,
    "cross":            SYMBOL_CROSS,
    "circle":           SYMBOL_CIRCLE,
}

# ══════════════════════════════════════════════════════════════════════════════
# Named colour palettes  (name → RGB)
# ══════════════════════════════════════════════════════════════════════════════

S_COLOR_MAP = {
    "green":       (0, 255, 0),
    "darkgreen":   (0, 200, 0),
    "lime":        (50, 205, 50),
    "forest":      (0, 128, 0),
    "lightgreen":  (144, 238, 144),
}
G_COLOR_MAP = {
    "blue":        (0, 0, 255),
    "red":         (255, 0, 0),
    "orange":      (255, 165, 0),
    "darkblue":    (0, 0, 200),
    "purple":      (128, 0, 128),
}

# ══════════════════════════════════════════════════════════════════════════════
# Defaults (fixed values when that factor is NOT being varied)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_S_SYMBOL = "arrow_right"    # start marker
DEFAULT_G_SYMBOL = "arrow_down"     # goal  marker
DEFAULT_SCALE    = 3                # pixel-art scale
DEFAULT_S_CLR    = "green"
DEFAULT_G_CLR    = "blue"

PIXEL_SCALES = [2, 3, 4, 5]

# ══════════════════════════════════════════════════════════════════════════════
# Symbol style pairings  (start_symbol, goal_symbol)
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL_STYLE_PAIRS = [
    ("arrow_right", "arrow_down"),
    ("arrow_up",    "arrow_left"),
    ("star",        "diamond"),
    ("triangle",    "circle"),
    ("cross",       "star"),
    ("diamond",     "triangle"),
    ("arrow_right", "star"),
    ("circle",      "cross"),
]

# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_pixel_symbol(draw, pattern, center, color, scale):
    """Render a 7×7 pixel-art bitmap centred at `center`."""
    cx, cy = center
    for dy, row in enumerate(pattern):
        for dx, val in enumerate(row):
            if val:
                x1 = cx + dx * scale
                y1 = cy + dy * scale
                x2 = x1 + scale - 1
                y2 = y1 + scale - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)


def get_symbol_bbox(scale):
    """Width and height of a 7×7 bitmap at the given scale."""
    return 7 * scale, 7 * scale

# ══════════════════════════════════════════════════════════════════════════════
# Free-space checks
# ══════════════════════════════════════════════════════════════════════════════

def is_free_pixel(pixel):
    return pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240


def is_region_free(image, x, y, w, h):
    img_w, img_h = image.size
    if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
        return False
    for dy in range(h):
        for dx in range(w):
            if not is_free_pixel(image.getpixel((x + dx, y + dy))):
                return False
    return True

# ══════════════════════════════════════════════════════════════════════════════
# Core: generate N samples for ONE specific config
# ══════════════════════════════════════════════════════════════════════════════

def generate_samples(maze_files, out_dir, s_symbol_name, g_symbol_name,
                     scale, s_color_name, g_color_name,
                     samples_per_maze=20):
    """
    For every maze image, generate `samples_per_maze` variations with
    the EXACT (start_symbol, goal_symbol, scale, colors) configuration.
    """
    os.makedirs(out_dir, exist_ok=True)
    s_rgb = S_COLOR_MAP[s_color_name]
    g_rgb = G_COLOR_MAP[g_color_name]
    s_pat = PATTERN_MAP[s_symbol_name]
    g_pat = PATTERN_MAP[g_symbol_name]
    s_w, s_h = get_symbol_bbox(scale)
    g_w, g_h = get_symbol_bbox(scale)

    total_saved = 0

    for image_path in maze_files:
        base_image = Image.open(image_path).convert("RGB")
        width, height = base_image.size
        name_only = os.path.splitext(os.path.basename(image_path))[0]

        margin = 5
        max_sx = width  - s_w - margin
        max_sy = height - s_h - margin
        max_gx = width  - g_w - margin
        max_gy = height - g_h - margin

        if max_sx <= margin or max_sy <= margin or max_gx <= margin or max_gy <= margin:
            continue

        for var_idx in range(1, samples_per_maze + 1):
            valid = False
            for _ in range(1000):
                sx = random.randint(margin, max_sx)
                sy = random.randint(margin, max_sy)
                gx = random.randint(margin, max_gx)
                gy = random.randint(margin, max_gy)

                if not is_region_free(base_image, sx, sy, s_w, s_h):
                    continue
                if not is_region_free(base_image, gx, gy, g_w, g_h):
                    continue
                # No overlap
                if abs(sx - gx) < max(s_w, g_w) and abs(sy - gy) < max(s_h, g_h):
                    continue
                valid = True
                break

            if not valid:
                continue

            img_copy = base_image.copy()
            draw = ImageDraw.Draw(img_copy)
            draw_pixel_symbol(draw, s_pat, (sx, sy), s_rgb, scale)
            draw_pixel_symbol(draw, g_pat, (gx, gy), g_rgb, scale)

            out_name = f"{name_only}_v{var_idx}.png"
            img_copy.save(os.path.join(out_dir, out_name))
            total_saved += 1

    return total_saved

# ══════════════════════════════════════════════════════════════════════════════
# Build the 4 ablation experiments
# ══════════════════════════════════════════════════════════════════════════════

def build_ablation_experiments():
    """
    Returns a list of experiment dicts:
        { group, subfolder, s_symbol, g_symbol, scale, s_color, g_color }
    """
    experiments = []

    # ── 1. Vary SYMBOL STYLE (fix scale, colors at defaults) ─────────────
    for s_sym, g_sym in SYMBOL_STYLE_PAIRS:
        experiments.append({
            "group":     "vary_symbol_style",
            "subfolder": f"{s_sym}__{g_sym}",
            "s_symbol":  s_sym,
            "g_symbol":  g_sym,
            "scale":     DEFAULT_SCALE,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 2. Vary SIZE (fix symbol=arrow_right/arrow_down, colors default) ─
    for sc in PIXEL_SCALES:
        experiments.append({
            "group":     "vary_size",
            "subfolder": f"scale{sc}",
            "s_symbol":  DEFAULT_S_SYMBOL,
            "g_symbol":  DEFAULT_G_SYMBOL,
            "scale":     sc,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 3. Vary S-COLOR (fix symbols, scale, G-color at defaults) ────────
    for cname in S_COLOR_MAP:
        experiments.append({
            "group":     "vary_s_color",
            "subfolder": f"s_{cname}",
            "s_symbol":  DEFAULT_S_SYMBOL,
            "g_symbol":  DEFAULT_G_SYMBOL,
            "scale":     DEFAULT_SCALE,
            "s_color":   cname,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 4. Vary G-COLOR (fix symbols, scale, S-color at defaults) ────────
    for cname in G_COLOR_MAP:
        experiments.append({
            "group":     "vary_g_color",
            "subfolder": f"g_{cname}",
            "s_symbol":  DEFAULT_S_SYMBOL,
            "g_symbol":  DEFAULT_G_SYMBOL,
            "scale":     DEFAULT_SCALE,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   cname,
        })

    return experiments

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def batch_process(input_dir, output_dir, samples_per_maze=20):
    maze_files = sorted(glob(os.path.join(input_dir, "*.png")))
    print(f"Found {len(maze_files)} maze files.\n")
    if not maze_files:
        print("No maze images found. Exiting.")
        return

    experiments = build_ablation_experiments()
    print(f"Total ablation experiments: {len(experiments)}\n")

    # ── CSV manifest ─────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.csv")
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    for exp in experiments:
        folder = os.path.join(output_dir, exp["group"], exp["subfolder"])
        print(f"▶ [{exp['group']}] {exp['subfolder']}  "
              f"(s={exp['s_symbol']}, g={exp['g_symbol']}, "
              f"scale={exp['scale']}, sc={exp['s_color']}, gc={exp['g_color']})")

        n = generate_samples(
            maze_files, folder,
            s_symbol_name=exp["s_symbol"],
            g_symbol_name=exp["g_symbol"],
            scale=exp["scale"],
            s_color_name=exp["s_color"],
            g_color_name=exp["g_color"],
            samples_per_maze=samples_per_maze,
        )
        print(f"   → saved {n} images to {folder}\n")

        manifest_rows.append({
            "group":      exp["group"],
            "subfolder":  exp["subfolder"],
            "s_symbol":   exp["s_symbol"],
            "g_symbol":   exp["g_symbol"],
            "scale":      exp["scale"],
            "s_color":    exp["s_color"],
            "g_color":    exp["g_color"],
            "num_images":  n,
            "path":        folder,
        })

    # Write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group", "subfolder", "s_symbol", "g_symbol", "scale",
            "s_color", "g_color", "num_images", "path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"📋 Manifest written to: {manifest_path}")
    print("🎉 Arrow/Symbol ablation dataset generation complete!")


if __name__ == "__main__":
    INPUT_FOLDER  = "/home/praveen_k/ddpm/pathddpm/testData/colored_mazes/"
    OUTPUT_FOLDER = "/home/praveen_k/mmNet/MazeWithArrows/"

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER, samples_per_maze=20)
