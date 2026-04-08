BEVFusion Setup and Patch Guide

This guide shows how to fully set up BEVFusion and apply my changes.


1. Install Requirements (Windows)

Install:

- Docker Desktop
- NVIDIA GPU Drivers
- NVIDIA Container Toolkit

Then open Windows PowerShell and verify:

in powershell:   docker --version

2. Clone BEVFusion (Windows PowerShell)

powershell:   git clone https://github.com/mit-han-lab/bevfusion.git
powershell:   cd bevfusion


3. Download Pretrained Model

Download `bevfusion-det.pth` and place it in:   bevfusion/pretrained/    (do this using File Explorer)

4. Start Docker Container (Windows PowerShell)

in powershell:

docker run -it `
  --gpus all `
  --name bevfusion_container `
  -v D:/your_data:/workspace/data `
  bevfusion_image bash

This opens a Linux terminal inside Docker.


5. Install BEVFusion (INSIDE Docker)

Now you are inside Docker (you’ll see root@...):

bash:

cd /workspace/bevfusion
pip install -v -e .


6. Add Your Dataset (Windows)

Place your data here using File Explorer:

text:

D:/your_data/my_bev_dataset/


Inside Docker, it appears as:

text:

/workspace/data/my_bev_dataset/


7. Apply Patch (INSIDE Docker)

Make sure your patch file is inside the repo, then run:

bash:

git apply bevfusion.patch


8. Run Evaluation (INSIDE Docker)

bash:

python3 tools/test.py \
  configs/custom/my_dataset.yaml \
  pretrained/bevfusion-det.pth \
  --eval bbox


Notes

Run PowerShell commands outside Docker
Run bash/python commands inside Docker
Paths must use `/workspace/data/...` inside Docker


Summary

PowerShell → clone + run Docker
Docker terminal → install + patch + evaluate
