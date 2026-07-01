"""""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel
from mmNet import ResNetVisualEncoder, FiLMGenerator, Config, PathUNet

class PolygonPathGenerator:
    def __init__(self,
                 world_grid_size=500,
                 model_grid_size=128,
                 resolution=0.05,
                 film_channels=32,
                 num_steps=200,
                 ckpt_path="/home/praveensingh/aed_environment/src/depth_to_pointcloud/src/scripts/radhaRamanOffice_55.pth",
                 device=None):

        self.world_size = world_grid_size
        self.model_size = model_grid_size
        self.resolution = resolution
        self.center_world = self.world_size // 2
        
        # Scaling factor: 500 / 128 = 3.90625
        self.scale_factor = float(self.world_size) / float(self.model_size)

        self.film_channels = film_channels
        self.num_steps = num_steps
        self.ckpt_path = ckpt_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False

        self.tf = transforms.Compose([
            transforms.Resize((self.model_size, self.model_size)),
            transforms.ToTensor()
        ])

    def world_to_pixel_worldgrid(self, x, y):
        px = int(x / self.resolution + self.center_world)
        py = int(y / self.resolution + self.center_world)
        return px, py

    def pixel_worldgrid_to_world(self, px, py):
        x = (px - self.center_world) * self.resolution
        y = (py - self.center_world) * self.resolution
        return x, y

    def generate_polygon_map(self, polygons_world, start, goal, filename):
        img = np.full((self.world_size, self.world_size, 3), 255, dtype=np.uint8)
        
        for poly in polygons_world:
            if len(poly) < 2: continue
            pts = [self.world_to_pixel_worldgrid(p[0], p[1]) for p in poly]
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, (0, 0, 0), 2)

        # Draw Start/Goal LARGE so they survive downsampling
        sx, sy = self.world_to_pixel_worldgrid(start[0], start[1])
        gx, gy = self.world_to_pixel_worldgrid(goal[0], goal[1])
        cv2.circle(img, (sx, sy), 15, (0, 255, 0), -1) 
        cv2.circle(img, (gx, gy), 15, (255, 0, 0), -1) 

        cv2.imwrite(filename, img)
        return filename

    def load_model(self):
        if self.model_loaded: return
        print(f"✔ Loading model from {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        
        self.unet = UNet2DModel(in_channels=3+self.film_channels, out_channels=1, layers_per_block=2, block_out_channels=(64, 128, 256, 512), norm_num_groups=32, attention_head_dim=8).to(self.device)
        self.encoder = ResNetVisualEncoder(output_dim=128).to(self.device)
        self.film_gen = FiLMGenerator(128, self.film_channels).to(self.device)
        
        self.unet.load_state_dict(ckpt["ema"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.film_gen.load_state_dict(ckpt["film"])
        
        self.unet.eval(); self.encoder.eval(); self.film_gen.eval()
        self.model_loaded = True

    @torch.no_grad()
    def generate_path_image(self, map_png, output_png):
        self.load_model()
        maze_img = Image.open(map_png).convert("RGB")
        maze = self.tf(maze_img).unsqueeze(0).to(self.device)
        features = self.encoder(maze)
        gamma, beta = self.film_gen(features)
        
        scheduler = DDPMScheduler(num_train_timesteps=self.num_steps, beta_schedule="squaredcos_cap_v2")
        scheduler.set_timesteps(self.num_steps, device=self.device)
        sample = torch.randn((1, 1, self.model_size, self.model_size), device=self.device)

        for t in scheduler.timesteps:
            modulated = gamma * sample.repeat(1, self.film_channels, 1, 1) + beta
            model_input = torch.cat([modulated, maze], dim=1)
            noise_pred = self.unet(model_input, t).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        out = sample.squeeze().cpu().numpy()
        out = np.clip(out, 0, 1) * 255
        Image.fromarray(out.astype(np.uint8)).save(output_png)
        return output_png

    # ------------------------------------------------------------------
    #  STEP 1: Extract & Clean Points in 128-Grid
    #  (Uses Local Mean to create a single thin line)
    # ------------------------------------------------------------------
    def extract_path_128(self, path_png, start_world, goal_world):
        img = cv2.imread(path_png, 0)
        if img is None: return []

        # Threshold
        _, mask = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        
        # Bridge Gaps (Dilation) to ensure connectivity
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Largest Blob Filter (Noise Removal)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels < 2: return []
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Get all pixels of the path
        ys, xs = np.where(labels == largest_label)
        cloud_points = np.column_stack((xs, ys)).astype(float)

        # Calculate Start/Goal in 128-Grid
        sx_500, sy_500 = self.world_to_pixel_worldgrid(start_world[0], start_world[1])
        gx_500, gy_500 = self.world_to_pixel_worldgrid(goal_world[0], goal_world[1])
        
        # Start/Goal in 128 coordinates
        start_128 = np.array([sx_500 / self.scale_factor, sy_500 / self.scale_factor])
        goal_128 = np.array([gx_500 / self.scale_factor, gy_500 / self.scale_factor])

        # --- SORTING WITH LOCAL MEAN (CENTERLINE) ---
        sorted_pixels = [start_128]
        current_pos = start_128
        
        available_mask = np.ones(len(cloud_points), dtype=bool)
        
        # Radius to average pixels (in 128-grid pixels)
        # 3 pixels in 128 grid is approx 12 pixels in 500 grid
        SEARCH_RADIUS = 3.0 

        while np.any(available_mask):
            # Find neighbors
            dists = np.linalg.norm(cloud_points - current_pos, axis=1)
            neighbors_mask = available_mask & (dists < SEARCH_RADIUS)
            
            if np.sum(neighbors_mask) > 0:
                # TAKE MEAN: This collapses the thick line into a single center point
                cluster = cloud_points[neighbors_mask]
                mean_pt = np.mean(cluster, axis=0)
                
                sorted_pixels.append(mean_pt)
                current_pos = mean_pt
                
                # Mark as visited
                available_mask[neighbors_mask] = False
            else:
                # Gap jump (Find closest remaining)
                remaining_indices = np.where(available_mask)[0]
                candidates = cloud_points[remaining_indices]
                
                dists_rem = np.linalg.norm(candidates - current_pos, axis=1)
                min_idx = np.argmin(dists_rem)
                
                if dists_rem[min_idx] > 10.0: # Gap too big? Stop.
                    break
                
                best_pt = candidates[min_idx]
                sorted_pixels.append(best_pt)
                current_pos = best_pt
                
                global_idx = remaining_indices[min_idx]
                available_mask[global_idx] = False
            
            # Snap to goal
            if np.linalg.norm(current_pos - goal_128) < 2.0:
                break

        sorted_pixels.append(goal_128)
        return sorted_pixels

    # ------------------------------------------------------------------
    #  MAIN ENTRYPOINT
    # ------------------------------------------------------------------
    def generate_path(self, polygons_world, start, goal,
                      out_map="polygon_map.png",
                      out_gen="generated_path.png",
                      out_vis=None):

        # 1. Generate Input
        self.generate_polygon_map(polygons_world, start, goal, out_map)
        # 2. Run Inference
        self.generate_path_image(out_map, out_gen)

        # 3. Extract Clean Centerline in 128-Grid
        path_128 = self.extract_path_128(out_gen, start, goal)
        
        if not path_128:
            print("⚠ No path found.")
            return [], out_map, out_gen

        # 4. Scale to 500-Grid and World Meters
        path_500 = []   # For Visualization (500x500)
        path_world = [] # For Robot (Meters)
        
        for p in path_128:
            # Scale coordinates (128 -> 500)
            px_500 = p[0] * self.scale_factor
            py_500 = p[1] * self.scale_factor
            path_500.append([int(px_500), int(py_500)])
            
            # Convert to Meters
            wx, wy = self.pixel_worldgrid_to_world(px_500, py_500)
            path_world.append([wx, wy])

        # 5. Save Visualization (500x500)
        if out_vis:
            try:
                vis_img = cv2.imread(out_map) # Load the clean map
                if vis_img is not None:
                    pts = np.array(path_500, dtype=np.int32)
                    
                    # CONNECT THE POINTS (This creates the smooth line)
                    cv2.polylines(vis_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                    
                    # Draw Start/Goal
                    cv2.circle(vis_img, tuple(pts[0]), 5, (0, 255, 0), -1)
                    cv2.circle(vis_img, tuple(pts[-1]), 5, (255, 0, 0), -1)
                    
                    cv2.imwrite(out_vis, vis_img)
                    print(f"✔ Visualization saved: {out_vis}")
            except Exception as e:
                print(f"⚠ Vis error: {e}")

        print(f"✔ Path generated: {len(path_world)} points")
        return path_world, out_map, out_gen

"""""
import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel
from mmNet import ResNetVisualEncoder, FiLMGenerator, Config, PathUNet

class PolygonPathGenerator:
    def __init__(self,
                 world_grid_size=500,
                 model_grid_size=128,
                 resolution=0.05,
                 film_channels=32,
                 num_steps=20,
                 ckpt_path="",
                 device=None):

        self.world_size = world_grid_size
        self.model_size = model_grid_size
        self.resolution = resolution
        self.center_world = self.world_size // 2
        self.scale_factor = float(self.world_size) / float(self.model_size)

        self.film_channels = film_channels
        self.num_steps = num_steps
        self.ckpt_path = ckpt_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False

        self.tf = transforms.Compose([
            transforms.Resize((self.model_size, self.model_size)),
            transforms.ToTensor()
        ])

    def world_to_pixel_worldgrid(self, x, y):
        px = int(x / self.resolution + self.center_world)
        py = int(y / self.resolution + self.center_world)
        return px, py

    def pixel_worldgrid_to_world(self, px, py):
        x = (px - self.center_world) * self.resolution
        y = (py - self.center_world) * self.resolution
        return x, y

    def load_model(self):
        if self.model_loaded: return
        print(f"✔ Loading model from {self.ckpt_path}")
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        
        self.unet = UNet2DModel(in_channels=3+self.film_channels, out_channels=1, layers_per_block=2, block_out_channels=(64, 128, 256, 512), norm_num_groups=32, attention_head_dim=8).to(self.device)
        self.encoder = ResNetVisualEncoder(output_dim=128).to(self.device)
        self.film_gen = FiLMGenerator(128, self.film_channels).to(self.device)
        
        self.unet.load_state_dict(ckpt["ema"])
        self.encoder.load_state_dict(ckpt["encoder"])
        self.film_gen.load_state_dict(ckpt["film"])
        
        self.unet.eval(); self.encoder.eval(); self.film_gen.eval()
        self.model_loaded = True

    @torch.no_grad()
    def generate_path_image(self, map_png, output_png):
        self.load_model()
        maze_img = Image.open(map_png).convert("RGB")
        maze = self.tf(maze_img).unsqueeze(0).to(self.device)
        features = self.encoder(maze)
        gamma, beta = self.film_gen(features)
        
        scheduler = DDPMScheduler(num_train_timesteps=self.num_steps, beta_schedule="squaredcos_cap_v2")
        scheduler.set_timesteps(self.num_steps, device=self.device)
        sample = torch.randn((1, 1, self.model_size, self.model_size), device=self.device)

        for t in scheduler.timesteps:
            modulated = gamma * sample.repeat(1, self.film_channels, 1, 1) + beta
            model_input = torch.cat([modulated, maze], dim=1)
            noise_pred = self.unet(model_input, t).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        out = sample.squeeze().cpu().numpy()
        out = np.clip(out, 0, 1) * 255
        Image.fromarray(out.astype(np.uint8)).save(output_png)
        return output_png

    def extract_path_128(self, path_png, start_world, goal_world):
        img = cv2.imread(path_png, 0)
        if img is None: return []

        _, mask = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels < 2: return []
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        ys, xs = np.where(labels == largest_label)
        cloud_points = np.column_stack((xs, ys)).astype(float)

        sx_500, sy_500 = self.world_to_pixel_worldgrid(start_world[0], start_world[1])
        gx_500, gy_500 = self.world_to_pixel_worldgrid(goal_world[0], goal_world[1])
        
        start_128 = np.array([sx_500 / self.scale_factor, sy_500 / self.scale_factor])
        goal_128 = np.array([gx_500 / self.scale_factor, gy_500 / self.scale_factor])

        sorted_pixels = [start_128]
        current_pos = start_128
        available_mask = np.ones(len(cloud_points), dtype=bool)
        
        SEARCH_RADIUS = 3.0 

        while np.any(available_mask):
            dists = np.linalg.norm(cloud_points - current_pos, axis=1)
            neighbors_mask = available_mask & (dists < SEARCH_RADIUS)
            
            if np.sum(neighbors_mask) > 0:
                cluster = cloud_points[neighbors_mask]
                mean_pt = np.mean(cluster, axis=0)
                sorted_pixels.append(mean_pt)
                current_pos = mean_pt
                available_mask[neighbors_mask] = False
            else:
                remaining_indices = np.where(available_mask)[0]
                candidates = cloud_points[remaining_indices]
                dists_rem = np.linalg.norm(candidates - current_pos, axis=1)
                min_idx = np.argmin(dists_rem)
                if dists_rem[min_idx] > 10.0: break
                best_pt = candidates[min_idx]
                sorted_pixels.append(best_pt)
                current_pos = best_pt
                available_mask[remaining_indices[min_idx]] = False
            
            if np.linalg.norm(current_pos - goal_128) < 2.0: break

        sorted_pixels.append(goal_128)
        return sorted_pixels

    def generate_path(self, start, goal, input_map_path="input_map.png", out_gen="generated_path.png", out_vis=None):
        if not os.path.exists(input_map_path):
            print(f"❌ Error: Input map not found: {input_map_path}")
            return [], input_map_path, out_gen

        self.generate_path_image(input_map_path, out_gen)
        path_128 = self.extract_path_128(out_gen, start, goal)
        
        if not path_128:
            print("⚠ No path found.")
            return [], input_map_path, out_gen

        path_world = [] 
        path_500 = []
        for p in path_128:
            px_500 = p[0] * self.scale_factor
            py_500 = p[1] * self.scale_factor
            path_500.append([int(px_500), int(py_500)])
            wx, wy = self.pixel_worldgrid_to_world(px_500, py_500)
            path_world.append([wx, wy])

        if out_vis:
            try:
                vis_img = cv2.imread(input_map_path)
                if vis_img is not None:
                    pts = np.array(path_500, dtype=np.int32)
                    cv2.polylines(vis_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
                    cv2.circle(vis_img, tuple(pts[0]), 5, (0, 255, 0), -1)
                    cv2.circle(vis_img, tuple(pts[-1]), 5, (255, 0, 0), -1)
                    cv2.imwrite(out_vis, vis_img)
            except Exception as e: print(f"⚠ Vis error: {e}")

        print(f"✔ Path generated: {len(path_world)} points")
        return path_world, input_map_path, out_gen