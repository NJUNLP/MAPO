from vllm import LLM, SamplingParams
import torch
import json
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, default="/mnt/data/shesj/PLM/MetaMath-7B-V1.0")
parser.add_argument('--testset',  type=str, default="/mnt/data/shesj/Data/RL4CoTData/gsm8k_test.json")
parser.add_argument('--suffix',  type=str, default="-gen")
parser.add_argument('--iter',  type=int, default=1)
parser.add_argument('--temp',  type=float, default=0.3)
args = parser.parse_args()
data_file = args.testset
print(args)
f = open(data_file,'r')
data = json.load(f)
if 'prompt' in data[0]:
    input_prompt = [i['prompt'] for i in data]
else:
    input_prompt = [i['instruction'] for i in data]

sampling_params = SamplingParams(n=args.iter,temperature=args.temp,max_tokens=512)
llm = LLM(model=args.model_path,dtype=torch.bfloat16,tensor_parallel_size=8)

generations = llm.generate(input_prompt,sampling_params)
generated_ = []
for output in generations:
    prompt = output.prompt
    generate_text = [o.text for o in output.outputs]
    generated_.append(generate_text)

assert len(data) == len(generated_)
for i,g in zip(data,generated_):
    if 'answers' not in i:
        i['answers'] = []
    for _ in g:
        i['answers'].append({"generated":_})

f = open(data_file.replace(".json","{}.json".format(args.suffix)),'w')
json.dump(data,f,indent=2,ensure_ascii=False)
