#!/bin/bash
model_name=$1
split=$2


# To accelerate, we will launch 8 processes on 8 GPUs simultaneously to calculate the scores. You can configure `num_proc` and `data_len = (total_data / num_proc)` according to your actual situation.

#SET here
#config the paired data path
datapath="numglue-mutli-lingualmeta_13B_genen_collect.json"

path1="../Data/tmp/$datapath"
echo path: "$path1"

if [ -d "$path1" ]; then
  rm -r "$path1"
fi

mkdir "$path1"

num_proc=8
data_len=2400
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

# Call json_merge to merge the results of each process
python json_merged.py --source_dir ../Data/tmp/$datapath --target_file ../Data/feedback_data/$datapath
