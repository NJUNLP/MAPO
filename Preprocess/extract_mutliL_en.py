
import sys
sys.path.append("/mnt/data/shesj/RL4CoT")
import json
import math

import json
from collections import Counter



def check_repeated_sentences(paragraph):
    # 使用换行符分割段落为句子
    paragraph = paragraph.replace('\n\n','\n')
    sentences = paragraph.split('\n')
    # 计算每个句子出现的次数
    sentence_counts = Counter(sentences)
    
    # 检查是否有句子出现次数超过3
    for sentence, count in sentence_counts.items():
        #print(sentence,count)
        if count >= 2:
            #print(f"句子 '{sentence}' 出现了 {count} 次。")
            #exit()
            return True
    return False


file="/mnt/data/shesj/Data/RL4CoTData/numglue-mutli-lingualmeta_13B_gen.json"

def preprocess(solution):
    solution =  solution.split("####")[0].strip()
    solution = re.sub(u"\\<<.*?\\>>", "", solution)
    return solution
import re

en_question2en_answer = {}

with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(data[0])
result = []
index = 0
add_count = 0
sampled_data = data
for item in (sampled_data):
    if 'prompt' in item:
        item['instruction'] = item['prompt']
    else:
        item['prompt'] = item['instruction']
    #del item['prompt']
    lang =  item['instruction'].split("Please answer in ")[1].split(".\n\n### Instruction:")[0]
    if 'en_question' not in item and "en_instruction" in item:
        item['en_question'] = item['en_instruction']
    if lang == "English":
        if 'en_question' not in item:
            item['en_question'] = item['instruction'].split("\n\n### Response:")[0].split("### Instruction:\n")[1].strip()
        temp = en_question2en_answer.get(item['en_question'],[])
        answers = item['answers']
        temp += [preprocess(_['generated']) for _ in answers]
        temp = list(set(temp))
        filtered_temp = []
        for t in temp:
            if check_repeated_sentences(t) == False:
                filtered_temp.append(t)
        temp = filtered_temp
        en_question2en_answer[item['en_question']] = temp
        add_count += 1

print(add_count)

avg_en = 0
count = 0

for item in (sampled_data):
    if 'prompt' in item:
        item['instruction'] = item['prompt']
    else:
        item['prompt'] = item['instruction']
    #del item['prompt']
    lang =  item['instruction'].split("Please answer in ")[1].split(".\n\n### Instruction:")[0]
    if lang != "English":
        en_solution = en_question2en_answer.get(item['en_question'],[])
        if len(en_solution) == 0:
            print("hit empty")
        item['en_collected_answer'] = en_solution
        avg_en += len(en_solution)
        count += 1

print(avg_en/count)
print(len(sampled_data))


file = file.replace(".json","en_collect.json")
with open(file, 'w', encoding='utf-8') as fw:
    json.dump(sampled_data, fw, indent=2,ensure_ascii=False)

