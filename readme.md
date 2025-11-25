# Feedforward 4D Reconstruction for driving scenes

# Install
```bash
conda create -n 4dv python=3.10 -y
conda activate 4dv
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

# Train
```bash
cd src
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu train.py --config-path ../config/waymo --config-name stage1_online
```

# Inference
```bash
./run_inference.sh 100 100 100000 
# Arguments: <start_frame> <end_frame> <ckpt_step>
```
