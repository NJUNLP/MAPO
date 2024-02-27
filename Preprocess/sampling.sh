#SET HERE
# Set the model path and testset 
# iter denotes the output number sampled for one question
# temp denotes the temperature for sampling
# We append the suffix to the output file to distinguish different sampling settings
python3 vllm_sampling.py --model_path Parallel_7B --testset ../Data/numglue-mutli-lingual.json --iter 20 --temp 0.8 --suffix Parallel_7B_gen_0.8_20