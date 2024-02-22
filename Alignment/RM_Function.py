import torch
import re
import sys
import torch.nn as nn
import json
from collections import Counter

sys.path.append("/mnt/data/shesj/RL4CoT")
from utils.generatePrompt import Prompter,get_prompter

loss_fn = nn.CrossEntropyLoss(reduction='none')

def check_repeated_sentences(paragraph):
    # 使用换行符分割段落为句子
    paragraph = paragraph.replace('\n\n','\n')
    sentences = paragraph.split('\n')
    sentence_counts = Counter(sentences)
    
    for sentence, count in sentence_counts.items():
        if count >= 2:
            return True
    return False

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


from transformers import  pipeline

import re
def MultiLigual_Alighment_reward_fuction(tokenizer,rm_model,outputs,labels=None):
    model = rm_model[0]
    tokenizer = rm_model[1]
    status = {}

    if check_repeated_sentences(outputs.split('### Response:')[1].strip().split("####")[0].strip()):
        status['avg-PPL'] = -1
        status['reward'] = -1
        return status
        
    outputs = [outputs for i in range(len(labels))]
    assert len(outputs) == len(labels)
    prediction = [output.split('### Response:')[1].strip().split("####")[0].strip() for output in outputs]
    prediction = [re.sub(u"\\<<.*?\\>>", "", p) for p in prediction] 
    labels = [re.sub(u"\\<<.*?\\>>", "", label).split("####")[0].strip() for label in labels]
    target_lang = 'eng_Latn'
    print('output',outputs[0])
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

    with torch.no_grad():
        output = model(**x, labels=y.input_ids)
        loss = output.loss
        results = []
        for i in range(output.logits.size(0)):
            pre = output.logits[i]
            lab = y.input_ids[i]
            result = loss_fn(pre.view(-1, output.logits.size(-1)), lab.view(-1)).mean().cpu().detach().numpy().tolist()
            results.append(result)

        result = sum(results)/len(results)
        status['avg-PPL'] = result
        status['reward'] = 1/result
    torch.cuda.empty_cache()
    return status

reward_function_market = {
    'ml_align' : MultiLigual_Alighment_reward_fuction,
}