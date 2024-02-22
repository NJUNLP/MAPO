from vllm import LLM, SamplingParams
import torch
import json
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, default="MetaMath-7B-V1.0")
parser.add_argument('--testset',  type=str, default="gsm8k_test.json")
parser.add_argument('--iter',  type=int, default=1)
parser.add_argument('--temp',  type=float, default=0.3)

args = parser.parse_args()
data_file = args.testset
model_path = args.model_path
data_path = data_file

if 'checkpoint' in model_path:
    model_name = model_path.split("/")[-2] + "-" + model_path.split("/")[-1]
else:
    model_name = model_path.split("/")[-1]

target_dir = "./Generations/{model}-{split}".format(model = model_name,split = data_path.split("/")[-1])

f = open(data_file,'r')
data = json.load(f)

input_prompt = [i['prompt'] for i in data]
sampling_params = SamplingParams(n=args.iter,temperature=0,max_tokens=1024)
print(args.model_path)
llm = LLM(model=args.model_path,dtype=torch.bfloat16,tensor_parallel_size=8)

generations = llm.generate(input_prompt,sampling_params)
generated_ = []
for output in generations:
    prompt = output.prompt
    generate_text = [o.text for o in output.outputs]
    generated_.append(generate_text)

assert len(data) == len(generated_)
for i,g in zip(data,generated_):
    i['answers'] = []
    for _ in g:
        i['answers'].append({"generated":_})

f = open(target_dir,'w')
json.dump(data,f,indent=2,ensure_ascii=False)
