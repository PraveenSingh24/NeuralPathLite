import os
import csv
import random
import itertools
from glob import glob
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────────────
# Pixel-art patterns (5×5 bitmaps)
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
PIXEL_ALT_S = [
    [0,1,1,1,0],
    [1,0,0,0,0],
    [0,1,1,1,0],
    [0,0,0,0,1],
    [0,1,1,1,0],
]
PIXEL_ALT_G = [
    [0,1,1,1,0],
    [1,0,0,0,0],
    [1,0,0,1,1],
    [1,0,0,0,1],
    [0,1,1,1,0],
]

# ──────────────────────────────────────────────────────────────────────────────
# Named color palettes  (name → RGB)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (fixed values when that factor is NOT being varied)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_FONT   = "pixel"        # font style
DEFAULT_SCALE  = 3              # pixel-art scale
DEFAULT_TTSIZE = 16             # truetype size (only used when font=truetype)
DEFAULT_S_CLR  = "green"        # S color name
DEFAULT_G_CLR  = "blue"         # G color name

# ──────────────────────────────────────────────────────────────────────────────
# Size options
# ──────────────────────────────────────────────────────────────────────────────
PIXEL_SCALES   = [2, 3, 4, 5]
TRUETYPE_SIZES = [12, 16, 20, 24]

# ──────────────────────────────────────────────────────────────────────────────
# TrueType font discovery
# ──────────────────────────────────────────────────────────────────────────────
FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

def find_available_fonts():
    """Return dict of short-name → path for available TrueType fonts."""
    name_map = {
        "DejaVuSans-Bold":      "DejaVuSans-Bold",
        "LiberationMono-Bold":  "LiberationMono-Bold",
        "FreeSansBold":         "FreeSansBold",
        "DejaVuSans":           "DejaVuSans",
        "LiberationSans-Bold":  "LiberationSans-Bold",
    }
    available = {}
    for fp in FONT_SEARCH_PATHS:
        base = os.path.splitext(os.path.basename(fp))[0]
        if os.path.isfile(fp) and base not in available:
            available[base] = fp
    return available

AVAILABLE_FONTS = find_available_fonts()

# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_pixel_letter(draw, pattern, center, color, scale):
    cx, cy = center
    for dy, row in enumerate(pattern):
        for dx, val in enumerate(row):
            if val:
                x1 = cx + dx * scale
                y1 = cy + dy * scale
                x2 = x1 + scale - 1
                y2 = y1 + scale - 1
                draw.rectangle([x1, y1, x2, y2], fill=color)

def get_pixel_bbox_size(scale):
    return 5 * scale, 5 * scale

def draw_truetype_letter(draw, letter, center, color, font):
    draw.text(center, letter, fill=color, font=font)

def get_truetype_bbox_size(letter, font):
    bbox = font.getbbox(letter)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

# ──────────────────────────────────────────────────────────────────────────────
# Free-space checks (full bounding-box)
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# Core: generate N samples for ONE specific config
# ──────────────────────────────────────────────────────────────────────────────

def get_font_patterns(font_name):
    """Return (S_pattern, G_pattern) for pixel fonts, or None for truetype."""
    if font_name == "pixel":
        return PIXEL_S, PIXEL_G
    elif font_name == "pixel_alt":
        return PIXEL_ALT_S, PIXEL_ALT_G
    return None  # truetype

def generate_samples(maze_files, out_dir, font_name, size, s_color_name,
                     g_color_name, samples_per_maze=20):
    """
    For every maze image, generate `samples_per_maze` variations with
    the EXACT given (font, size, s_color, g_color) configuration.
    """
    os.makedirs(out_dir, exist_ok=True)
    s_rgb = S_COLOR_MAP[s_color_name]
    g_rgb = G_COLOR_MAP[g_color_name]

    is_tt = font_name not in ("pixel", "pixel_alt")
    if is_tt:
        font_path = AVAILABLE_FONTS.get(font_name)
        if font_path is None:
            print(f"  ⚠️  TrueType font '{font_name}' not found, skipping.")
            return 0
        font_obj = ImageFont.truetype(font_path, size)
        s_w, s_h = get_truetype_bbox_size("S", font_obj)
        g_w, g_h = get_truetype_bbox_size("G", font_obj)
    else:
        patterns = get_font_patterns(font_name)
        s_w, s_h = get_pixel_bbox_size(size)
        g_w, g_h = get_pixel_bbox_size(size)

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
                if abs(sx - gx) < max(s_w, g_w) and abs(sy - gy) < max(s_h, g_h):
                    continue
                valid = True
                break

            if not valid:
                continue

            img_copy = base_image.copy()
            draw = ImageDraw.Draw(img_copy)

            if is_tt:
                draw_truetype_letter(draw, "S", (sx, sy), s_rgb, font_obj)
                draw_truetype_letter(draw, "G", (gx, gy), g_rgb, font_obj)
            else:
                s_pat, g_pat = patterns
                draw_pixel_letter(draw, s_pat, (sx, sy), s_rgb, size)
                draw_pixel_letter(draw, g_pat, (gx, gy), g_rgb, size)

            out_name = f"{name_only}_v{var_idx}.png"
            img_copy.save(os.path.join(out_dir, out_name))
            total_saved += 1

    return total_saved

# ──────────────────────────────────────────────────────────────────────────────
# Build the 4 ablation experiments
# ──────────────────────────────────────────────────────────────────────────────

def build_ablation_experiments():
    """
    Returns a list of experiment dicts:
        { group, subfolder, font, size, s_color, g_color }
    """
    experiments = []

    # ── 1. Vary FONT (fix size, S-color, G-color at defaults) ────────────
    pixel_fonts = ["pixel", "pixel_alt"]
    tt_fonts    = list(AVAILABLE_FONTS.keys())
    all_fonts   = pixel_fonts + tt_fonts

    for fname in all_fonts:
        sz = DEFAULT_TTSIZE if fname not in pixel_fonts else DEFAULT_SCALE
        experiments.append({
            "group":     "vary_font",
            "subfolder": fname,
            "font":      fname,
            "size":      sz,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 2. Vary SIZE (fix font=pixel, S-color, G-color at defaults) ─────
    for sc in PIXEL_SCALES:
        experiments.append({
            "group":     "vary_size",
            "subfolder": f"scale{sc}",
            "font":      "pixel",
            "size":      sc,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 3. Vary S-COLOR (fix font=pixel, size=default, G-color=default) ──
    for cname in S_COLOR_MAP:
        experiments.append({
            "group":     "vary_s_color",
            "subfolder": f"s_{cname}",
            "font":      "pixel",
            "size":      DEFAULT_SCALE,
            "s_color":   cname,
            "g_color":   DEFAULT_G_CLR,
        })

    # ── 4. Vary G-COLOR (fix font=pixel, size=default, S-color=default) ──
    for cname in G_COLOR_MAP:
        experiments.append({
            "group":     "vary_g_color",
            "subfolder": f"g_{cname}",
            "font":      "pixel",
            "size":      DEFAULT_SCALE,
            "s_color":   DEFAULT_S_CLR,
            "g_color":   cname,
        })

    return experiments

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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
              f"(font={exp['font']}, size={exp['size']}, "
              f"s={exp['s_color']}, g={exp['g_color']})")

        n = generate_samples(
            maze_files, folder,
            font_name=exp["font"],
            size=exp["size"],
            s_color_name=exp["s_color"],
            g_color_name=exp["g_color"],
            samples_per_maze=samples_per_maze,
        )
        print(f"   → saved {n} images to {folder}\n")

        manifest_rows.append({
            "group":     exp["group"],
            "subfolder": exp["subfolder"],
            "font":      exp["font"],
            "size":      exp["size"],
            "s_color":   exp["s_color"],
            "g_color":   exp["g_color"],
            "num_images": n,
            "path":      folder,
        })

    # Write manifest
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group", "subfolder", "font", "size",
            "s_color", "g_color", "num_images", "path",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"📋 Manifest written to: {manifest_path}")
    print("🎉 Ablation dataset generation complete!")


if __name__ == "__main__":
    INPUT_FOLDER  = "/home/praveen_k/ddpm/pathddpm/testData/colored_mazes/"
    OUTPUT_FOLDER = "/home/praveen_k/mmNet/MazeWithSG/"

    batch_process(INPUT_FOLDER, OUTPUT_FOLDER, samples_per_maze=20)