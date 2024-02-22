MODEL_PATH="/mnt/data/shesj/PLM/MetaMath-7B-V1.0"
python3 vllm_test.py --model_path $MODEL_PATH --testset ./Benchmarks/mgsm8k_test_genen_collect.json --iter 1 --temp 0
python3 vllm_test.py --model_path $MODEL_PATH --testset ./Benchmarks/MSVAMP_test_genen_collect.json --iter 1 --temp 0
python3 vllm_test.py --model_path $MODEL_PATH --testset ./Benchmarks/NumGLUE_test_multi_genen_collect.json --iter 1 --temp 0





