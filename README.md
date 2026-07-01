# NeuralPathLite: A Lightweight Image-Conditioned Diffusion Planner for Autonomous Mobile Robots

This repository contains the official implementation of **NeuralPathLite**, a lightweight neural-network-based, image-conditioned diffusion planner optimized for embedded and centralized swarm architectures. The framework features a Semantic Visual Anchoring mechanism to prevent mode collapse in cluttered environments and is validated on both the Mobile Robot Local Planning Benchmark (MRPB) and a physical Clearpath Husky A200 platform.

## 🛠️ System Requirements & Prerequisites
Before setting up the workspace, ensure your system meets the following requirements:
* **Operating System:** Ubuntu 20.04 LTS
* **Middleware:** ROS Noetic
* **Storage:** Git Large File Storage (Git LFS) must be installed to pull the 3D environment meshes and datasets.

Install core external dependencies:
```bash
sudo apt-get install ros-noetic-desktop-full git-lfs
pip3 install torch torchvision torchaudio numpy


🚀 Installation & Setup

    Clone the repository with Git LFS support:
    Bash

git clone [https://github.com/PraveenSingh24/NeuralPathLite.git](https://github.com/PraveenSingh24/NeuralPathLite.git)
cd NeuralPathLite

Initialize Git LFS to pull the large mesh and data assets (campus.dae, SimData/, etc.):
Bash

git lfs pull

Build the Catkin workspace:
Bash

    cd src/.. # Navigate to your workspace root if applicable
    catkin_make
    source devel/setup.bash

📂 Repository Structure

    src/ - Core ROS nodes, custom launch files, and the Husky navigation integration stack.

    src/vehicle_simulator/ - 3D simulation environments (including the campus mesh and garage point clouds).

    SimData/ - Conditioned trajectory datasets, pre-trained diffusion model weights, and benchmark logs.

🏃 How to Run
1. Launch the Simulation Environment

To spin up the gazebo/rviz visualization environment with the target map:
Bash

roslaunch vehicle_simulator system_simulation.launch

2. Execute the NeuralPathLite Diffusion Planner

In a new terminal, run the image-conditioned diffusion planning node:
Bash

rosrun neural_path_planning planner_node.py

📝 Citation

If you find our work or dataset useful in your research, please cite our IEEE T-ASE paper:
Code snippet

@article{kumar2026neuralpathlite,
  title={NeuralPathLite: A Lightweight Image-Conditioned Diffusion Planner for Efficient and Safe Navigation of Service Robots},
  author={Kumar, Praveen and Gupta, Pratham and Sandhan, Tushar},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  publisher={IEEE}
}
