OpenPCDet + PointPillar (KITTI) Setup Guide

This guide explains how to fully install, set up, and run the PointPillar pretrained model using the KITTI 3D Object Detection dataset.


# 1. Install WSL (Windows Only)

Open Windows PowerShell (Admin):

wsl --install
```

Restart your PC.


2. Install NVIDIA Drivers + CUDA

1. Install latest NVIDIA driver (from NVIDIA website)
2. Install CUDA Toolkit (match PyTorch compatibility)

Check GPU:

nvidia-smi


3. Install Basic Dependencies (WSL Ubuntu)

sudo apt update
sudo apt install -y git python3-pip python3-dev build-essential


4. Clone OpenPCDet

git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet


5. Create Python Environment

python3 -m venv opcdet_env
source opcdet_env/bin/activate
pip install --upgrade pip


6. Install PyTorch (GPU Example)

Check your CUDA version, then install matching PyTorch:

Example (CUDA 11.8):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118



7. Install OpenPCDet Requirements

pip install -r requirements.txt
python setup.py develop



 8. Download KITTI Dataset

Download from official KITTI website:

Required files:

- Velodyne point clouds
- Labels
- Calibration files
- Images


9. Organize Dataset

Inside OpenPCDet:

OpenPCDet/
└── data/
    └── kitti/
        ├── ImageSets/
        ├── training/
        │   ├── velodyne/
        │   ├── label_2/
        │   ├── calib/
        │   ├── image_2/
        └── testing/


10. Generate KITTI Info Files

python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml


11. Apply Custom Patch

Place the patch file in root:

OpenPCDet.patch

Apply it:

git apply OpenPCDet.patch


12. Replace GPU IoU with CPU Version

Copy your file:

rotate_iou_cpu.py

Make sure it is located here:

pcdet/datasets/kitti/kitti_object_eval_python/

This replaces GPU dependency for IoU calculation.


13. (IMPORTANT) Fix Dataset Path

Open:

tools/cfgs/dataset_configs/kitti_dataset.yaml

Set:

DATA_PATH: data/kitti


# 14. Download Pretrained Model

Download PointPillar pretrained weights and place in:

OpenPCDet/pretrained/

Example:

pretrained/pointpillar_7728.pth


15. Run Evaluation

From OpenPCDet root:

python tools/test.py \
  --cfg_file tools/cfgs/kitti_models/pointpillar.yaml \
  --ckpt pretrained/pointpillar_7728.pth \
  --eval_tag pointpillar_pretrained


16. (Optional) Save Predictions

--save_to_file

Common Issues


Dataset shows 0 samples

Check `DATA_PATH`
Regenerate `.pkl` files

CUDA errors

Use CPU IoU file (already included)
Ensure PyTorch CUDA matches system CUDA

Permission issues (WSL)

Avoid using Windows mounted drives for training
Keep project inside Linux filesystem

Notes

No training is required (pretrained model used)
Main steps: install → dataset → infos → run
Always verify paths carefully
