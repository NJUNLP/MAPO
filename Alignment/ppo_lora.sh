CONFIG=$1
PORT=$(( $RANDOM % 1000 + 32768 ))
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export GRUB_CMDLINE_LINUX_DEFAULT="iommu=soft"
deepspeed --master_port $PORT ppo.py --training_config $CONFIG  > /mnt/data/shesj/print_log/$1
