#!/bin/bash
model_name=$1
split=$2

datapath="numglue-mutli-lingualmeta_13B_genen_collect.json"

path1="/mnt/data/shesj/Data/RL4CoTData/tmp/$datapath"
echo path: "$path1"

if [ -d "$path1" ]; then
  rm -r "$path1"
fi

mkdir "$path1"

num_proc=8
#data_len=3800
data_len=2280
data_len=2400
# data_len=700
#data_len=10
processes=()

for ((i = 0; i < num_proc; i++)); do
    begin_index=$((i * data_len))
    CUDA_VISIBLE_DEVICES=$i python PPL_batched.py.py --begin_index "$begin_index" --data_length "$data_len" --data_file "$datapath" &
    echo "Process $i Begin!"
    processes+=($!)
done

# 等待所有进程完成
for pid in "${processes[@]}"; do
    wait "$pid"
done

echo "所有进程已完成"

echo "Merging..."

python json_merged.py --source_dir /mnt/data/shesj/Data/RL4CoTData/tmp/$datapath --target_file /mnt/data/shesj/Data/RL4CoTData/feedback_data/$datapath
