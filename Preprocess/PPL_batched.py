from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig
import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from random import sample
import sys
sys.path.append("/mnt/data/shesj/RL4CoT")
from utils.generatePrompt import get_prompter,parse_answer,parse_reasoning,get_split
import math
import argparse
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss(reduction='none')
parser = argparse.ArgumentParser(description="sampling argument")
parser.add_argument('--begin_index', type=int)
parser.add_argument('--data_length', type=int)
parser.add_argument('--data_file', type=str)
args = parser.parse_args()
data_path = "/mnt/data/shesj/Data/RL4CoTData/{}".format(args.data_file)
target_dir = "/mnt/data/shesj/Data/RL4CoTData/tmp/{}".format(args.data_file)
target_path = target_dir + "/{proc}.json".format(proc = args.begin_index)


data = []
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
rm_model_base = AutoModelForSeq2SeqLM.from_pretrained("/mnt/data/shesj/PLM/nllb-200-distilled-600M",device_map='auto')
rm_model_base = rm_model_base.eval()
rm_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/shesj/PLM/nllb-200-distilled-600M")
rm_model = (rm_model_base,rm_tokenizer)

from transformers import  pipeline

import re
def MultiLigual_Alighment_reward_fuction(tokenizer,rm_model,outputs,labels=None):
    model = rm_model[0]
    tokenizer = rm_model[1]
    prediction = [output.split('### Response:')[1].strip().split("####")[0].strip() for output in outputs]
    prediction = [re.sub(u"\\<<.*?\\>>", "", p) for p in prediction] 
    labels = [re.sub(u"\\<<.*?\\>>", "", label).split("####")[0].strip() for label in labels]
    target_lang = 'eng_Latn'
    l = outputs[0].split("Please answer in ")[1].split(".\n\n### Instruction:")[0]
    langs = {
        'Swahili' :'swh_Latn',
        'Chinese' : "zho_Hans",
        "Bengali" : "ben_Beng",
        "German" : "deu_Latn",
        "Spanish" : "spa_Latn",
        "French" : "fra_Latn",
        "Japanese" : "jpn_Hani",
        "Russian" : "rus_Cyrl",
        "Thai" : "tha_Thai",
        "English" : "eng_Latn"
        } 
    
    target_lang = 'eng_Latn'   
    source_lang = langs[l]
    tokenizer.src_lang = source_lang

    x = tokenizer(prediction, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
    tokenizer.src_lang = target_lang
    y = tokenizer(labels, return_tensors='pt', padding='longest', truncation=True,max_length=512).to(model.device)
    y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
    results = []
    with torch.no_grad():
        output = model(**x, labels=y.input_ids)
        loss = output.loss

        for i in range(output.logits.size(0)):
            pre = output.logits[i]
            lab = y.input_ids[i]
            result = loss_fn(pre.view(-1, output.logits.size(-1)), lab.view(-1)).mean().cpu().detach().numpy().tolist()
            results.append(1/result)
    
    torch.cuda.empty_cache()
    return results


begin_index = args.begin_index
data_length = args.data_length
end_index = min(len(data), begin_index + data_length)
print("begin_index: {}, end_index: {}".format(begin_index, end_index))

result = []
for i in tqdm(range(begin_index, end_index)):
    item = data[i]
    lang =  item['prompt'].split("Please answer in ")[1].split(".\n\n### Instruction:")[0]
    if lang == "English":
        continue
    if len(item['en_collected_answer']) == 0:
        continue

    for it in item['answers']: 
        reward_list = []
        input_answer = [item['prompt'] + it['generated'] for j in item['en_collected_answer']]
        output = [j for j in item['en_collected_answer']]
        if len(input_answer) <= 20:
            reward_list = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model,input_answer,output)
        else:
            reward_list1 = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model,input_answer[:20],output[:20])
            reward_list2 = MultiLigual_Alighment_reward_fuction(rm_tokenizer,rm_model,input_answer[20:],output[20:])
            reward_list = reward_list1 + reward_list2
            assert len(reward_list) == len(output)
            
        torch.cuda.empty_cache()
        it['nllb-200-distilled-600M-reward-mean'] =  sum(reward_list)/len(reward_list)
        it['nllb-200-distilled-600M-reward-max'] =  max(reward_list)
        it['nllb-200-distilled-600M-reawrdlist'] =  reward_list
    result.append(item)

with open(target_path, 'w', encoding='utf-8') as fw:
    json.dump(result, fw, indent=2, ensure_ascii=False)
        
        
        



