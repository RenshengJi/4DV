# Feedforward 4D Reconstruction for Driving Scenes

## Install

```bash
conda create -n 4dv python=3.10 -y
conda activate 4dv
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
```

## Dataset

Dataset: https://huggingface.co/datasets/renshengjihe/waymo-flow-seg

```bash
python datasets_preprocess/extract_tars.py ../data/waymo/tar --num-workers 64
```

## Train

```bash
cd src
CUDA_VISIBLE_DEVICES=0 accelerate launch --multi_gpu train.py --config-path ../config/waymo --config-name train 
```

## Inference

生成包含GT/Pred对比的可视化：RGB、Depth、Velocity、Segmentationn、ynamic Clustering

```bash
CUDA_VISIBLE_DEVICES=0 python inference_multi.py --config-path ./config/waymo --config-name infer_multi 
```

