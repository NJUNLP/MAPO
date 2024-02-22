CONFIG=$1
PORT=$(( $RANDOM % 1000 + 32768 ))
# do not remove or the training will hang and nodes will be lost w/o this workaround
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
export GRUB_CMDLINE_LINUX_DEFAULT="iommu=soft"
#deepspeed --master_port $PORT rebuild_LLMtrainer.py --train_config $CONFIG > "/mnt/data/sheshuaijie/LocalOutput/Log/$CONFIG.log"
#deepspeed --master_port $PORT mGPU_dev_ppo_rm.py --reward_fuction "upper-token"
#deepspeed --master_port $PORT mGPU_dev_ppo_rm_dev.py --reward_fuction "upper-token"
#deepspeed --master_port $PORT mGPU_dev_ppo_rm_dev.py --reward_fuction "more-step"
#CUDA_VISIBLE_DEVICES=0,5,6,7 accelerate launch --config_file="/mnt/data/shesj/RL4CoT/PPO/deepspeed_zero2.yaml" --num_processes 4 dpo.py --training_config $CONFIG > /mnt/data/shesj/print_log/$1
#accelerate launch --config_file="/mnt/data/shesj/RL4CoT/PPO/deepspeed_zero2.yaml" --num_processes 4 dpo.py --training_config $CONFIG > /mnt/data/shesj/print_log/$1
accelerate launch --config_file="/mnt/data/shesj/RL4CoT/PPO/deepspeed_zero2_dpo.yaml" --num_processes 8 dpo_lora.py --training_config $CONFIG > /mnt/data/shesj/print_log/$1
