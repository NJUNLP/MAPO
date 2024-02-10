MODEL_PATH=$1
python3 vllm_test.py --model_path $MODEL_PATH --testset ../Data/mgsm8k_test_genen_collect.json --iter 1 --temp 0
python3 vllm_test.py --model_path $MODEL_PATH --testset ../MSVAMP_test_genen_collect.json --iter 1 --temp 0
python3 vllm_test.py --model_path $MODEL_PATH --testset ../NumGLUE_test_multi_genen_collect.json --iter 1 --temp 0






