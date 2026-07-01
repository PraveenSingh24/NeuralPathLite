"""""
import os
import numpy as np
from PIL import Image
import random

# Configurations
RESOLUTION = 128
NUM_RECT_OBSTACLES = 8
NUM_CIRC_OBSTACLES = 2
OUTPUT_FOLDER = "testData/custom_maze"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def add_rectangular_obstacle(maze, top_left, width, height, value=0):
    x, y = top_left
    x_end = min(x + width, maze.shape[1])
    y_end = min(y + height, maze.shape[0])
    maze[y:y_end, x:x_end] = value
    return maze

def add_circular_obstacle(maze, center, radius, value=0):
    Y, X = np.ogrid[:maze.shape[0], :maze.shape[1]]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    mask = dist <= radius ** 2
    maze[mask] = value
    return maze

def generate_custom_maze(resolution):
    maze = np.ones((resolution, resolution), dtype=np.uint8)

    # Add random rectangular obstacles
    for _ in range(NUM_RECT_OBSTACLES):
        w = random.randint(10, 25)
        h = random.randint(2, 10)
        x = random.randint(0, resolution - w)
        y = random.randint(0, resolution - h)
        maze = add_rectangular_obstacle(maze, (x, y), w, h, value=0)

    # Add random circular obstacles
    for _ in range(NUM_CIRC_OBSTACLES):
        r = random.randint(5, 25)
        cx = random.randint(r, resolution - r)
        cy = random.randint(r, resolution - r)
        maze = add_circular_obstacle(maze, (cx, cy), r, value=0)

    return maze

def save_maze_as_image(maze, filename):
    img = Image.fromarray((maze * 255).astype(np.uint8))  # Binary to grayscale
    img.save(filename)

def main():
    maze = generate_custom_maze(RESOLUTION)
    save_maze_as_image(maze, os.path.join(OUTPUT_FOLDER, "custom_maze.png"))
    print("✅ Maze saved to 'custom_maze.png'")

if __name__ == "__main__":
    main()

import numpy as np
from PIL import Image
import random

def generate_and_save_maze():
    height, width = 128, 128
    maze = np.ones((height, width), dtype=np.uint8)  # 1 = free space

    row = 64  # place both obstacles on row 64

    # Calculate max col1 so both obstacles + 5px gap fit in image
    #max_col1 = width - (50 + 5 + 10)
    #col1 = random.randint(10, max_col1)
    #col2 = col1 + 50 + 5

    # Draw first obstacle: 1 row high, 50 cols wide
    maze[row, 30:60] = 0

    # Draw second obstacle: 1 row high, 10 cols wide
    maze[row, 65:110] = 0

    # Save image: 1 -> white (255), 0 -> black (0)
    img = Image.fromarray((maze * 255).astype(np.uint8))  # ✅ Don't invert here
    img.save("generated_maze.png")

generate_and_save_maze()


from PIL import Image
import numpy as np
import os

def separate_maze_and_points(input_image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load RGB image
    img = Image.open(input_image_path).convert("RGB")
    img_np = np.array(img)

    # Initialize blank RGB image for points (black background)
    points_only = np.ones_like(img_np)*255

    # Create masks for start (green) and goal (blue)
    green_mask = np.all(img_np == [0, 255, 0], axis=-1)
    blue_mask = np.all(img_np == [0, 0, 255], axis=-1)

    # Extract start and goal points into the RGB image
    points_only[green_mask] = [0, 255, 0]
    points_only[blue_mask] = [0, 0, 255]

    # Remove the start/goal colors from the maze (set those pixels to white)
    cleaned_maze = img_np.copy()
    cleaned_maze[green_mask | blue_mask] = [255, 255, 255]  # set to white
    maze_gray = Image.fromarray(np.array(Image.fromarray(cleaned_maze).convert("L")))

    # Save outputs
    maze_gray.save(os.path.join(output_dir, "maze_gray.png"))
    Image.fromarray(points_only).save(os.path.join(output_dir, "start_goal_points.png"))
    print("✅ Saved: maze_gray.png and start_goal_points.png")

if __name__ == "__main__":
    input_path = "../ddpm/output_sets/maze/maze_100_vis_0.png"  # <-- replace with your actual input
    output_directory = "output_split"
    separate_maze_and_points(input_path, output_directory)


# generate_noise.py

import torch
import torchvision.transforms as T
from PIL import Image

# Define the noise size
height, width = 128, 128

# Generate Gaussian noise (mean=0, std=1)
noise_tensor = torch.randn(1, height, width)  # Single-channel noise

# Normalize to [0, 255] for saving as image
noise_normalized = (noise_tensor - noise_tensor.min()) / (noise_tensor.max() - noise_tensor.min())
noise_image = T.ToPILImage()(noise_normalized)

# Save the noise image
noise_image.save("noise_128x128.png")

print("Noise image saved as noise_128x128.png")
"""
import os
import numpy as np
from PIL import Image
import random

# Configurations
RESOLUTION = 128
NUM_RECT_OBSTACLES = 4
NUM_CIRC_OBSTACLES = 3
OUTPUT_FOLDER = "testData/T3Colored_maze"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Colors
BACKGROUND_COLOR = [255, 255, 255]  # White
RECT_COLOR = [255, 105, 180]         # Pink (Hot Pink)
CIRC_COLOR = [255, 165, 0]           # Orange

def add_rectangular_obstacle(maze, top_left, width, height):
    x, y = top_left
    x_end = min(x + width, maze.shape[1])
    y_end = min(y + height, maze.shape[0])
    maze[y:y_end, x:x_end] = RECT_COLOR
    return maze

def add_circular_obstacle(maze, center, radius):
    Y, X = np.ogrid[:maze.shape[0], :maze.shape[1]]
    dist = (X - center[0]) ** 2 + (Y - center[1]) ** 2
    mask = dist <= radius ** 2
    maze[mask] = CIRC_COLOR
    return maze

def generate_custom_maze(resolution):
    # Create a white background maze
    maze = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255

    # Add random rectangular obstacles (pink)
    for _ in range(NUM_RECT_OBSTACLES):
        w = random.randint(10, 25)
        h = random.randint(2, 10)
        x = random.randint(0, resolution - w)
        y = random.randint(0, resolution - h)
        maze = add_rectangular_obstacle(maze, (x, y), w, h)

    # Add random circular obstacles (orange)
    for _ in range(NUM_CIRC_OBSTACLES):
        r = random.randint(5, 25)
        cx = random.randint(r, resolution - r)
        cy = random.randint(r, resolution - r)
        maze = add_circular_obstacle(maze, (cx, cy), r)

    return maze

def save_maze_as_image(maze, filename):
    img = Image.fromarray(maze)
    img.save(filename)

def main():
    for idx in range(600):  # Generate 10 mazes
        maze = generate_custom_maze(RESOLUTION)
        filename = os.path.join(OUTPUT_FOLDER, f"custom_maze_{idx}.png")
        save_maze_as_image(maze, filename)
        print(f"✅ Maze {idx} saved to '{filename}'")

if __name__ == "__main__":
    main()


from PIL import Image, ImageEnhance, ImageFilter
import os

def resize_and_enhance(input_path, output_path, original_size=(180, 180),
                       sharpen_factor=2.5, contrast_factor=1.3, edge_enhance=True):

    if not os.path.exists(input_path):
        print(f"❌ Input file does not exist: {input_path}")
        return

    try:
        # Load and convert image
        img = Image.open(input_path).convert("RGB")

        # Resize using high-quality resampling
        img = img.resize(original_size, resample=Image.LANCZOS)

        # Enhance contrast
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(contrast_factor)

        # Enhance sharpness
        sharpener = ImageEnhance.Sharpness(img)
        img = sharpener.enhance(sharpen_factor)

        # Optional: apply edge enhancement
        if edge_enhance:
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)

        # Save result
        img.save(output_path, format="PNG")
        print(f"✅ Enhanced image saved: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    input_image = "../ddpm/testResults/mrpb_output/sparse_0_vis_209.png"
    output_image = "../ddpm/testResults/mrpb_output/original_0_vis_209_enhanced.png"

    resize_and_enhance(
        input_path=input_image,
        output_path=output_image,
        original_size=(180, 180),
        sharpen_factor=2.5,
        contrast_factor=1.3,
        edge_enhance=True
    )


import os
import numpy as np
import random
from PIL import Image
import heapq

class MazeSolver:
    def __init__(self):
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def a_star(self, start, goal, maze):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            visited.add(current)

            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid_move(maze, neighbor) or neighbor in visited:
                    continue

                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def is_valid_move(self, maze, pos):
        x, y = pos
        return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] == 0

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

def load_maze_from_image(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((64, 64), resample=Image.NEAREST)
    maze = np.array(img)
    maze = (maze > 127).astype(np.uint8)  # 0: free, 1: wall
    return maze

def draw_maze_with_path_and_points(maze, path, start, goal):
    img = np.stack([maze * 255] * 3, axis=-1).astype(np.uint8)

    for x, y in path:
        img[x, y] = [255, 0, 0]  # Red path

    img[start[0], start[1]] = [0, 255, 0]  # Green start
    img[goal[0], goal[1]] = [0, 0, 255]   # Blue goal

    return Image.fromarray(img)

def create_path_image(maze, path):
    path_img = np.zeros_like(maze)
    for x, y in path:
        path_img[x, y] = 1
    return path_img

def generate_start_and_goal(maze, min_distance=10, max_distance=100):
    free = np.argwhere(maze == 0)
    if len(free) < 2:
        return None, None

    for _ in range(100):
        start = tuple(free[random.randint(0, len(free) - 1)])
        goal = tuple(free[random.randint(0, len(free) - 1)])
        dist = np.linalg.norm(np.array(start) - np.array(goal))
        if min_distance <= dist <= max_distance:
            return start, goal
    return None, None

def is_valid_point(maze, point):
    x, y = point
    return 0 <= x < maze.shape[0] and 0 <= y < maze.shape[1] and maze[x, y] == 0

def has_free_neighbors(maze, point, radius=1):
    x, y = point
    h, w = maze.shape
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and maze[nx, ny] == 1:
                return False
    return True

def main():
    input_maze_folder = "/home/praveen_k/ddpm/pathddpm/testData/maze_radharani"
    out_base = "/home/praveen_k/mmNet/testData/input_set_radhaRani"
    os.makedirs(out_base, exist_ok=True)

    path_img_dir = os.path.join(out_base, "paths_only")
    vis_img_dir = os.path.join(out_base, "maze_with_paths")
    sg_txt_dir = os.path.join(out_base, "start_goal_txts")
    radius_txt_dir = os.path.join(out_base, "radius_txts")

    os.makedirs(path_img_dir, exist_ok=True)
    os.makedirs(vis_img_dir, exist_ok=True)
    os.makedirs(sg_txt_dir, exist_ok=True)
    os.makedirs(radius_txt_dir, exist_ok=True)

    maze_files = [f for f in os.listdir(input_maze_folder) if f.endswith(".png")]

    for maze_file in maze_files:
        print(f"Processing {maze_file}")
        maze_path = os.path.join(input_maze_folder, maze_file)
        maze = load_maze_from_image(maze_path)
        maze_id = os.path.splitext(maze_file)[0]
        count = 0

        while count < 10:
            radius = random.randint(0, 5)
            maze_solver = MazeSolver()

            def valid(p):
                return is_valid_point(maze, p) and has_free_neighbors(maze, p, radius)

            start, goal = generate_start_and_goal(maze, min_distance=30, max_distance=120)
            if start is None or goal is None or not valid(start) or not valid(goal):
                continue

            path = maze_solver.a_star(start, goal, maze)

            if not path:
                continue

            path_img = create_path_image(maze, path)
            Image.fromarray((path_img * 255).astype(np.uint8)).save(
                os.path.join(path_img_dir, f"{maze_id}_path_{count}.png"))

            vis_img = draw_maze_with_path_and_points(maze, path, start, goal)
            vis_img.save(os.path.join(vis_img_dir, f"{maze_id}_vis_{count}.png"))

            with open(os.path.join(sg_txt_dir, f"{maze_id}_sg_{count}.txt"), "w") as f:
                f.write(f"Start: {start}\nGoal: {goal}\n")

            with open(os.path.join(radius_txt_dir, f"{maze_id}_radius_{count}.txt"), "w") as f:
                f.write(f"{radius}\n")

            print(f"  ✔ Saved path {count+1} with radius {radius}")
            count += 1

if __name__ == "__main__":
    main()

from PIL import Image, ImageDraw, ImageFont
import os

letters = ['A', 'B', 'C', 'D', 'E', 'F']
colors = ['red', 'blue', 'green', 'purple', 'magenta', 'gold']
img_size = (128, 128)
font_size = 100  # Increase this to make letters bigger

# Load a truetype font (DejaVuSans is often bundled with Pillow)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
except IOError:
    font = ImageFont.load_default()
    print("Default font loaded, custom font not found.")

output_dir = "letter_images"
os.makedirs(output_dir, exist_ok=True)

for letter, color in zip(letters, colors):
    img = Image.new("RGB", img_size, "white")
    draw = ImageDraw.Draw(img)
    
    # Get text bounding box to center text
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
    draw.text(position, letter, font=font, fill=color)
    
    img.save(os.path.join(output_dir, f"{letter}.png"))



radius = random.randint(1, 5)
start, goal = generate_start_and_goal(maze)
#start, goal = [(65, 100), (65, 5)]
if start is None or goal is None:
    continue
path = maze_solver.a_star(start, goal, maze, radius)
if not path:
    continue
#smoothed_path = smooth_path_b_spline(path)
#path_img = create_path_image(maze, smoothed_path)
path_img = create_path_image(maze, path)
Image.fromarray((path_img * 255).astype(np.uint8)).save(
    os.path.join(path_img_dir, f"{maze_id}_path_{count}.png"))
#vis_img = draw_maze_with_path_and_points(maze, smoothed_path, start, goal)
vis_img = draw_maze_with_path_and_points(maze_color, path, start, goal)
vis_img.save(os.path.join(vis_img_dir, f"{maze_id}_vis_{count}.png"))
with open(os.path.join(text_dir, f"{maze_id}_sg_{count}.txt"), "w") as f:
    f.write(f"Start Point: {start}\\nGoal Point: {goal}\\n")

with open(os.path.join(txt_dir, f"{maze_id}_radius_{count}.txt"), "w") as f:
    f.write(f"Start Point: {radius}\\n")
count += 1
print(f"  ✔ Pair {count} done")

@torch.no_grad()
def validate(model, encoder, film_gen, val_loader, scheduler, device):
    model.eval()
    total_mse = 0
    for batch in val_loader:
        maze = batch['maze'].to(device)
        path = batch['path'].to(device)
        letter = batch['letter'].to(device).unsqueeze(1)
        encoder_input = torch.cat([maze, letter], dim=1)
        maze_features = encoder(encoder_input)
        #conditioned_features = torch.cat([maze_features, radius], dim=1)
        gamma, beta = film_gen(maze_features)

        noise = torch.randn_like(path)
        timesteps = torch.randint(0, config.num_timesteps, (path.size(0),), device=device)
        noisy_path = scheduler.add_noise(path, noise, timesteps)

        modulated = gamma * noisy_path.repeat(1, config.film_channels, 1, 1) + beta
        model_input = torch.cat([modulated, maze], dim=1)

        noise_pred = model(model_input, timesteps).sample
        mse = F.mse_loss(noise_pred, noise)
        total_mse += mse.item()
    
    avg_val_loss = total_mse / len(val_loader)
    print(f"📊 Validation MSE: {avg_val_loss:.6f}")
    return avg_val_loss
"""""
