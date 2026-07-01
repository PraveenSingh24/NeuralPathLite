import os
import random
from glob import glob
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────────────────────
# Pixel-art patterns for D and R (5×5 bitmaps)
# ──────────────────────────────────────────────────────────────────────────────
PIXEL_D = [
    [1,1,1,1,0],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,0,0,0,1],
    [1,1,1,1,0],
]
PIXEL_R = [
    [1,1,1,1,0],
    [1,0,0,0,1],
    [1,1,1,1,0],
    [1,0,1,0,0],
    [1,0,0,1,0],
]

# ── Base defaults (same as ablation default) ──
SCALE    = 3
D_COLOR  = (0, 255, 0)    # green  (same as S default)
R_COLOR  = (0, 0, 255)    # blue   (same as G default)

# ──────────────────────────────────────────────────────────────────────────────
# Drawing & free-space helpers
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


def is_region_free(image, x, y, w, h):
    img_w, img_h = image.size
    if x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h:
        return False
    for dy in range(h):
        for dx in range(w):
            px = image.getpixel((x + dx, y + dy))
            if not (px[0] > 240 and px[1] > 240 and px[2] > 240):
                return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_DR_mazes(input_dir, output_dir, samples_per_maze=20):
    os.makedirs(output_dir, exist_ok=True)
    maze_files = sorted(glob(os.path.join(input_dir, "*.png")))
    print(f"Found {len(maze_files)} maze files.\n")

    letter_w = 5 * SCALE
    letter_h = 5 * SCALE
    margin = 5
    total = 0

    for image_path in maze_files:
        filename = os.path.basename(image_path)
        name_only = os.path.splitext(filename)[0]

        base_image = Image.open(image_path).convert("RGB")
        width, height = base_image.size

        max_x = width  - letter_w - margin
        max_y = height - letter_h - margin
        if max_x <= margin or max_y <= margin:
            print(f"⚠️  Image too small: {filename}")
            continue

        for var in range(1, samples_per_maze + 1):
            valid = False
            for _ in range(1000):
                dx = random.randint(margin, max_x)
                dy_ = random.randint(margin, max_y)
                rx = random.randint(margin, max_x)
                ry = random.randint(margin, max_y)

                if not is_region_free(base_image, dx, dy_, letter_w, letter_h):
                    continue
                if not is_region_free(base_image, rx, ry, letter_w, letter_h):
                    continue
                if abs(dx - rx) < letter_w and abs(dy_ - ry) < letter_h:
                    continue
                valid = True
                break

            if not valid:
                continue

            img = base_image.copy()
            draw = ImageDraw.Draw(img)
            draw_pixel_letter(draw, PIXEL_D, (dx, dy_), D_COLOR, SCALE)
            draw_pixel_letter(draw, PIXEL_R, (rx, ry), R_COLOR, SCALE)

            img.save(os.path.join(output_dir, f"{name_only}_v{var}.png"))
            total += 1

    print(f"🎉 Done! Saved {total} images to {output_dir}")


if __name__ == "__main__":
    INPUT_FOLDER  = "/home/praveen_k/ddpm/pathddpm/testData/colored_mazes/"
    OUTPUT_FOLDER = "/home/praveen_k/mmNet/MazeWithDR/"

    generate_DR_mazes(INPUT_FOLDER, OUTPUT_FOLDER, samples_per_maze=20)
