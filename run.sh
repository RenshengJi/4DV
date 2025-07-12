#!/bin/bash

i=220500  # 初始索引值

while true; do
    echo "Processing index: $i"
    # Run the demo_video.py script with the specified model path and index
    python demo_video.py \
        --model_path "/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(fix_mask+nometric+fixgs+depth+fixlpips)/checkpoint-epoch_1_11390.pth" \
        --output_dir "/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/results" \
        --idx "$i" > /dev/null 2>&1
    # python demo_video.py \
    #     --model_path "/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/src/checkpoints/waymo/step2(fix_mask+nometric+fixgs)/checkpoint-epoch_0_44268.pth" \
    #     --output_dir "/mnt/teams/algo-teams/yuxue.yang/4DVideo/ziqi/4DVideo/results2" \
    #     --idx "$i" > /dev/null 2>&1
    # 增加索引值
    i=$((i + 500))
done