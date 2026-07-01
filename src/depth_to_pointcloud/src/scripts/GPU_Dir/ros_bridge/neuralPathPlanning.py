"""""
import os
import time
import json
import numpy as np
from neuralPath import PolygonPathGenerator

SHARED_DIR = "/home/praveen_k/ros_bridge/shared/"

# Create a folder specifically for history/logs if you want
HISTORY_DIR = os.path.join(SHARED_DIR, "history")
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

processor = PolygonPathGenerator(
    world_grid_size=500,
    model_grid_size=128,
    resolution=0.05,
    ckpt_path="/home/praveen_k/mmNet/radhaRamanOffice_55.pth"
)

print("🚀 Inference daemon started on Desktop-B (Saving History)")

while True:
    task_file = os.path.join(SHARED_DIR, "task.json")
    result_file_tmp = os.path.join(SHARED_DIR, "result.json.tmp")
    result_file = os.path.join(SHARED_DIR, "result.json")

    if os.path.exists(task_file):
        try:
            time.sleep(0.05) # Wait for write to finish

            with open(task_file, "r") as f:
                data = json.load(f)

            task_id = data.get("task_id", str(int(time.time())))
            print(f"📥 Processing task {task_id}...")

            polygons = data["polygons"]
            start = data["start"]
            goal = data["goal"]

            # --- [NEW] Generate Unique Filename for this Task ---
            # We save it in the 'history' folder so your shared folder stays clean
            unique_vis_name = os.path.join(HISTORY_DIR, f"path_{task_id}.png")

            # Run Generator
            world_path, _, _ = processor.generate_path(
                polygons, start, goal,
                out_map=os.path.join(SHARED_DIR, "map.png"), # Overwritten every time (internal use)
                out_gen=os.path.join(SHARED_DIR, "gen.png"), # Overwritten every time (internal use)
                out_vis=unique_vis_name # <--- SAVES UNIQUE IMAGE HERE
            )

            # Save Result
            with open(result_file_tmp, "w") as f:
                json.dump({
                    "task_id": task_id,
                    "path_world": world_path
                }, f)
            os.rename(result_file_tmp, result_file)

            print(f"   [Saved] Visualization: {unique_vis_name}")
            print(f"📤 Result sent. Points: {len(world_path)}")

        except Exception as e:
            print(f"❌ Error: {e}")
        
        if os.path.exists(task_file):
            os.remove(task_file)

    time.sleep(0.05)
"""""
import os
import time
import json
import shutil
import numpy as np
from neuralPath import PolygonPathGenerator

# ---------------- CONFIGURATION ----------------
SHARED_DIR = "/home/praveen_k/ros_bridge/shared/"
CKPT_PATH = "/home/praveen_k/mmNet/radhaRamanOffice_55.pth"
# -----------------------------------------------

HISTORY_DIR = os.path.join(SHARED_DIR, "history")
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

processor = PolygonPathGenerator(
    world_grid_size=500,
    model_grid_size=128,
    resolution=0.05,
    ckpt_path=CKPT_PATH
)

print("🚀 Inference daemon started on Desktop-B (Robust Mode)")

while True:
    task_file = os.path.join(SHARED_DIR, "task.json")
    result_file_tmp = os.path.join(SHARED_DIR, "result.json.tmp")
    result_file = os.path.join(SHARED_DIR, "result.json")

    if os.path.exists(task_file):
        try:
            time.sleep(0.05) # Brief wait for write to finish

            # 1. READ DATA
            with open(task_file, "r") as f:
                data = json.load(f)

            # 2. [CRITICAL FIX] DELETE TASK FILE IMMEDIATELY
            # We delete it now so we don't accidentally delete a NEW task 
            # that arrives while we are processing this one.
            os.remove(task_file)

            # 3. PROCESS
            task_id = data.get("task_id", str(int(time.time())))
            map_filename = data.get("map_image_file", "input_map.png")
            full_map_path = os.path.join(SHARED_DIR, map_filename)
            start_world = data["start"]
            goal_world = data["goal"]

            print(f"📥 Processing Task {task_id}")

            if not os.path.exists(full_map_path):
                print(f"❌ Map missing: {full_map_path}")
                continue

            unique_vis_name = os.path.join(HISTORY_DIR, f"path_{task_id}.png")

            # Run Generator
            world_path, _, _ = processor.generate_path(
                start_world, goal_world,
                input_map_path=full_map_path, 
                out_gen=os.path.join(SHARED_DIR, "gen.png"),
                out_vis=unique_vis_name
            )

            # 4. SAVE RESULT
            with open(result_file_tmp, "w") as f:
                json.dump({
                    "task_id": task_id,
                    "path_world": world_path
                }, f)
            
            os.rename(result_file_tmp, result_file)
            print(f"📤 Sent result for Task {task_id}")

        except Exception as e:
            print(f"❌ Error: {e}")
            # If we failed to read, try to clean up so we don't loop forever
            if os.path.exists(task_file):
                os.remove(task_file)

    time.sleep(0.05)