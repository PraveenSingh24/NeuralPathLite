import os
import numpy as np
from PIL import Image
from glob import glob

# ──────────────────────────────────────────────────────────────────────────────
# Core merge function (same logic as original combineMazePath.py)
# ──────────────────────────────────────────────────────────────────────────────

def merge_maze_and_path(maze_path, path_path, output_path):
    """Overlay path (red) onto maze image and save."""
    maze = Image.open(maze_path).convert("RGB").resize((128, 128), Image.LANCZOS)
    path = Image.open(path_path).convert("L").resize((128, 128), Image.LANCZOS)

    maze_np = np.array(maze)
    path_np = np.array(path)

    # Threshold to find path pixels
    path_mask = path_np >= 50
    path_points = np.column_stack(np.where(path_mask))

    if len(path_points) < 2:
        return False  # not enough path pixels

    # Draw red path with small thickness
    for y, x in path_points:
        for dy in range(-1, 0):
            for dx in range(-1, 0):
                ny, nx = y + dy, x + dx
                if 0 <= ny < maze_np.shape[0] and 0 <= nx < maze_np.shape[1]:
                    maze_np[ny, nx] = [255, 0, 0]

    Image.fromarray(maze_np).save(output_path, format="PNG")
    return True


def merge_folder(maze_dir, path_dir, output_dir):
    """
    Merge all maze-path pairs in a single subfolder.
    Files are matched by SAME filename (batchInference preserves names).
    """
    os.makedirs(output_dir, exist_ok=True)
    maze_files = sorted(glob(os.path.join(maze_dir, "*.png")))
    merged = 0
    skipped = 0

    for maze_filepath in maze_files:
        fname = os.path.basename(maze_filepath)
        path_filepath = os.path.join(path_dir, fname)

        # Check if the path file exists, or try the 'gen_' renamed version
        if not os.path.exists(path_filepath):
            gen_fname = fname.replace("maze", "gen")
            path_filepath = os.path.join(path_dir, gen_fname)

        if not os.path.exists(path_filepath):
            skipped += 1
            continue

        out_filepath = os.path.join(output_dir, fname)
        try:
            if merge_maze_and_path(maze_filepath, path_filepath, out_filepath):
                merged += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ❌ Error merging {fname}: {e}")
            skipped += 1

    return merged, skipped


# ──────────────────────────────────────────────────────────────────────────────
# Batch: walk the ablation folder tree
# ──────────────────────────────────────────────────────────────────────────────

def batch_combine(maze_root, path_root, output_root):
    """
    Recursively walk `maze_root`. For every leaf folder with *.png files,
    merge with the matching folder under `path_root` and save to
    `output_root`.  Works for any folder depth.
    """
    total_merged  = 0
    total_skipped = 0

    for dirpath, dirnames, filenames in os.walk(maze_root):
        png_files = [f for f in filenames if f.lower().endswith(".png")]
        if not png_files:
            continue

        rel = os.path.relpath(dirpath, maze_root)
        p_dir = os.path.join(path_root, rel) if rel != "." else path_root
        o_dir = os.path.join(output_root, rel) if rel != "." else output_root

        if not os.path.isdir(p_dir):
            print(f"⚠️  Path dir missing: {p_dir} — skipping")
            continue

        print(f"▶ {rel}  ({len(png_files)} images)")
        m, s = merge_folder(dirpath, p_dir, o_dir)
        print(f"   → merged {m}, skipped {s}\n")

        total_merged  += m
        total_skipped += s

    print(f"🎉 Done!  Total merged: {total_merged}, skipped: {total_skipped}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MAZE_ROOT   = "/home/praveen_k/mmNet/script/circle_rect_val_dataset/"
    PATH_ROOT   = "/home/praveen_k/mmNet/Pout2/"
    OUTPUT_ROOT = "/home/praveen_k/mmNet/PMazePath/"

    batch_combine(MAZE_ROOT, PATH_ROOT, OUTPUT_ROOT)