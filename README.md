# 🚗 Capstone Object Detection — LiDAR-Based 3D Object Detection

> **Part of a larger capstone project** combining LiDAR object detection, multi-frame point cloud registration, and multi-camera projection into a full autonomous perception pipeline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Models Covered](#-models-covered)
- [Setup Guide — PointPillar (KITTI)](#-setup-guide--pointpillar-kitti)
- [Setup Guide — BEVFusion](#-setup-guide--bevfusion)
- [Custom Dataset & Annotations](#-custom-dataset--annotations)
- [Custom Patches](#-custom-patches)
- [Troubleshooting](#-troubleshooting)
- [Related Repositories](#-related-repositories)

---

## 🔭 Project Overview

This repository contains the **3D object detection** component of a capstone perception system. The goal is to detect and localize 3D objects (vehicles, pedestrians, etc.) from LiDAR point cloud data using two state-of-the-art deep learning models:

| Model | Framework | Data Format | Hardware |
|---|---|---|---|
| **PointPillar** | OpenPCDet | KITTI | GPU (or CPU fallback) |
| **BEVFusion** | MIT BEVFusion | Custom / nuScenes-style | GPU via Docker |

Both models use **pretrained weights** — no training from scratch is required. The workflow is: install → organize dataset → apply patches → run evaluation.

This module feeds into the broader capstone pipeline alongside:
- **Multi-frame point cloud registration** — aligning point clouds across time frames
- **Multi-camera projection** — mapping 3D detections back into camera image space

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────┐
│                  Sensor Inputs                      │
│         LiDAR Point Cloud  +  Camera Images         │
└────────────────┬──────────────────┬────────────────┘
                 │                  │
                 ▼                  ▼
   ┌─────────────────────┐   ┌──────────────────────┐
   │  Multi-Frame Point  │   │  Multi-Camera        │
   │  Cloud Registration │   │  Projection          │
   │  (mtyramsey repo)   │   │  (mtyramsey repo)    │
   └──────────┬──────────┘   └──────────┬───────────┘
              │                         │
              └────────────┬────────────┘
                           ▼
          ┌─────────────────────────────────┐
          │   3D Object Detection           │
          │   PointPillar  |  BEVFusion     │
          │       (THIS REPOSITORY)         │
          └─────────────────────────────────┘
                           │
                           ▼
               3D Bounding Box Predictions
              (class, position, size, heading)
```

---

## 📁 Repository Structure

```
Capstone_Object_Detection/
│
├── PointPillar(KITTI)README.txt      # Full setup guide for PointPillar on KITTI
├── README_BevFusion_Setup.txt        # Full setup guide for BEVFusion via Docker
│
├── OpenPCDet.patch                   # Custom patch applied to OpenPCDet framework
├── bevfusion.patch                   # Custom patch applied to BEVFusion framework
│
├── rotate_iou_cpu.py                 # CPU-based IoU replacement (no CUDA required)
│
└── task_1_annotations_2026_03_31_22_45_07_kitti raw format 1.0.zip
                                      # Custom annotated dataset in KITTI format
```

---

## 🤖 Models Covered

### PointPillar (via OpenPCDet)

**PointPillar** is a fast and accurate 3D object detection model that encodes LiDAR point clouds into vertical columns called "pillars," then uses a 2D CNN backbone for detection. It is evaluated here on the **KITTI 3D Object Detection** benchmark.

- Framework: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- Dataset: [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- Pretrained weights: `pointpillar_7728.pth`
- No GPU IoU required — replaced with CPU implementation (`rotate_iou_cpu.py`)

### BEVFusion (via MIT BEVFusion)

**BEVFusion** is a multi-modal 3D detection framework that fuses LiDAR and camera features in Bird's-Eye View (BEV) space. This repo uses it for detection evaluation on a custom dataset.

- Framework: [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion)
- Pretrained weights: `bevfusion-det.pth`
- Runs inside a **Docker container** with GPU passthrough

---

## ⚙️ Setup Guide — PointPillar (KITTI)

### Prerequisites

- Ubuntu (or WSL on Windows)
- NVIDIA GPU + drivers
- CUDA Toolkit (match your PyTorch version)
- Python 3.8+

### Step 1 — Install WSL (Windows Only)

```powershell
wsl --install
```
Restart your PC after installation.

### Step 2 — Install NVIDIA Drivers and CUDA

1. Install the latest NVIDIA driver from [nvidia.com](https://www.nvidia.com/drivers)
2. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) matching your PyTorch version
3. Verify with:
```bash
nvidia-smi
```

### Step 3 — Install System Dependencies

```bash
sudo apt update
sudo apt install -y git python3-pip python3-dev build-essential
```

### Step 4 — Clone OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
```

### Step 5 — Create a Python Virtual Environment

```bash
python3 -m venv opcdet_env
source opcdet_env/bin/activate
pip install --upgrade pip
```

### Step 6 — Install PyTorch (GPU)

Find the right command for your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/). Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 7 — Install OpenPCDet

```bash
pip install -r requirements.txt
python setup.py develop
```

### Step 8 — Download the KITTI Dataset

Download from the [official KITTI website](https://www.cvlibs.net/datasets/kitti/). You need:
- Velodyne point clouds
- Labels (`label_2`)
- Calibration files (`calib`)
- Left color images (`image_2`)

### Step 9 — Organize the Dataset

Place the downloaded files into the following structure inside your `OpenPCDet` directory:

```
OpenPCDet/
└── data/
    └── kitti/
        ├── ImageSets/
        ├── training/
        │   ├── velodyne/
        │   ├── label_2/
        │   ├── calib/
        │   └── image_2/
        └── testing/
```

### Step 10 — Generate KITTI Info Files

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \
    tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Step 11 — Apply the Custom Patch

Copy `OpenPCDet.patch` from this repository into the root of your `OpenPCDet` folder, then run:

```bash
git apply OpenPCDet.patch
```

### Step 12 — Install the CPU IoU Module

Copy `rotate_iou_cpu.py` into:

```
pcdet/datasets/kitti/kitti_object_eval_python/
```

This replaces the GPU-dependent IoU calculation with a CPU version, removing the CUDA requirement for evaluation.

### Step 13 — Fix the Dataset Path

Open `tools/cfgs/dataset_configs/kitti_dataset.yaml` and set:

```yaml
DATA_PATH: data/kitti
```

### Step 14 — Download the Pretrained Model

Download the PointPillar pretrained weights and place them at:

```
OpenPCDet/pretrained/pointpillar_7728.pth
```

### Step 15 — Run Evaluation

```bash
python tools/test.py \
    --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
    --ckpt pretrained/pointpillar_7728.pth \
    --eval_tag pointpillar_pretrained
```

To also save predictions to file, add `--save_to_file`.

---

## ⚙️ Setup Guide — BEVFusion

### Prerequisites

- Windows (with PowerShell) or Linux
- NVIDIA GPU + drivers
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Step 1 — Install Requirements

Install Docker Desktop and the NVIDIA Container Toolkit. Verify Docker works:

```powershell
docker --version
```

### Step 2 — Clone BEVFusion

```powershell
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion
```

### Step 3 — Download the Pretrained Model

Download `bevfusion-det.pth` and place it in `bevfusion/pretrained/` using File Explorer.

### Step 4 — Start the Docker Container

```powershell
docker run -it `
    --gpus all `
    --name bevfusion_container `
    -v D:/your_data:/workspace/data `
    bevfusion_image bash
```

> ⚠️ Replace `D:/your_data` with the path to your actual dataset folder. Inside Docker, this appears as `/workspace/data/`.

### Step 5 — Install BEVFusion Inside Docker

Once inside the Docker container (you'll see a `root@...` prompt):

```bash
cd /workspace/bevfusion
pip install -v -e .
```

### Step 6 — Add Your Dataset

Place your dataset in the folder you mounted (e.g. `D:/your_data/my_bev_dataset/`). Inside Docker it appears as `/workspace/data/my_bev_dataset/`.

### Step 7 — Apply the Custom Patch

```bash
git apply bevfusion.patch
```

### Step 8 — Run Evaluation

```bash
python3 tools/test.py \
    configs/custom/my_dataset.yaml \
    pretrained/bevfusion-det.pth \
    --eval bbox
```

> **Note:** PowerShell commands are run **outside** Docker. All `bash` / `python` commands are run **inside** Docker. Paths inside Docker must use `/workspace/data/...`.

---

## 🏷 Custom Dataset & Annotations

This repo includes a custom annotated dataset in KITTI format:

**File:** `task_1_annotations_2026_03_31_22_45_07_kitti raw format 1.0.zip`

This dataset was annotated using a labeling tool and exported in **KITTI raw format**, making it directly compatible with the OpenPCDet PointPillar pipeline. To use it:

1. Unzip the file
2. Merge with your KITTI dataset structure under `data/kitti/training/`
3. Re-run the info file generation (Step 10 above)

---

## 🔧 Custom Patches

Two patch files are included that modify the upstream frameworks for this project's specific needs:

| Patch | Target | Purpose |
|---|---|---|
| `OpenPCDet.patch` | OpenPCDet | Project-specific modifications to the detection pipeline |
| `bevfusion.patch` | MIT BEVFusion | Adaptations for the custom dataset and configuration |

Apply each patch to the root of its respective cloned repository using `git apply <patch_file>`.

---

## 🐛 Troubleshooting

### Dataset shows 0 samples
- Check that `DATA_PATH` in `kitti_dataset.yaml` is set to `data/kitti`
- Delete and regenerate the `.pkl` info files (Step 10)
- Verify the folder structure matches exactly what is shown in Step 9

### CUDA / GPU errors
- Use the CPU IoU file (`rotate_iou_cpu.py`) — already included
- Ensure your PyTorch CUDA version matches the system CUDA version (`nvidia-smi` vs `nvcc --version`)

### WSL path / permission issues
- Avoid placing your project inside Windows-mounted drives (e.g. `/mnt/c/...`)
- Keep the entire project inside the Linux filesystem (e.g. `~/OpenPCDet/`)

### Docker issues (BEVFusion)
- Ensure the NVIDIA Container Toolkit is properly installed and Docker can see your GPU (`docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`)
- Make sure the volume mount path exists before running the container

---

## 🔗 Related Repositories

This object detection module is one part of a larger capstone perception system. The other components are maintained separately:

| Repository | Description |
|---|---|
| [mtyramsey/Multiframe-pointcloud-registeration-and-multi-camera-projection](https://github.com/mtyramsey/Multiframe-pointcloud-registeration-and-multi-camera-projection) | Multi-frame LiDAR point cloud registration across time, and projection of 3D data into multi-camera image space |
| [EddieHebert/Capstone_Object_Detection](https://github.com/EddieHebert/Capstone_Object_Detection) | *(This repo)* 3D object detection using PointPillar and BEVFusion |

Together these modules form a complete pipeline: raw sensor data → aligned point clouds → camera-projected scene → detected 3D objects.

---

## 📚 References

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) — Open source toolbox for 3D object detection from point clouds
- [PointPillar Paper](https://arxiv.org/abs/1812.05784) — Lang et al., 2019
- [MIT BEVFusion](https://github.com/mit-han-lab/bevfusion) — Liu et al., 2022
- [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/) — Geiger et al., 2012
