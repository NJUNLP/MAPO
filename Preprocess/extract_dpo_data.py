import json
import re

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

target = "numglue-mutli-lingualmeta_13B_genen_collect_reset"

f = open("/mnt/data/shesj/Data/RL4CoTData/feedback_data/{}.json".format(target))
data = json.load(f)

def process(input_str):
    input_str = re.sub(u"\\<<.*?\\>>", "", input_str)
    return input_str

dpo_data = []

english_instruction2data = {}

for i in data:
    lang =  i['prompt'].split("Please answer in ")[1].split(".\n\n### Instruction:")[0]
    if 'answer' in i:
        label = extract_last_num(i['answer'])
    else:
        label = extract_last_num(i['chosen'])
    if lang != "English":

        sorted_output = [g for g in sorted(i['answers'],key=lambda x:x["nllb-200-distilled-600M-reward-mean"],reverse=True)]

        temp = english_instruction2data.get(i['en_question'],[])
        for j in range(len(sorted_output)-1):
            predict_answer = extract_last_num(sorted_output[j]['generated'])
            if abs(label - predict_answer) > 1e-3:
                continue
            for l in range(j+1,len(sorted_output)):
                sample = {}
                sample['accept'] = i['prompt'] + sorted_output[j]['generated']
                sample['reject'] = i['prompt'] + sorted_output[l]['generated']
                sample['score-diff'] = sorted_output[j]['nllb-200-distilled-600M-reward-mean']-sorted_output[l]['nllb-200-distilled-600M-reward-mean']
                if sorted_output[j]['nllb-200-distilled-600M-reward-mean'] != sorted_output[l]['nllb-200-distilled-600M-reward-mean'] and process(sorted_output[j]['generated']) != process(sorted_output[l]['generated']):
                    temp.append(sample)
        english_instruction2data[i['en_question']] = temp


ratio = 10
index = 0
train_data = []
dev_data = []
for i in english_instruction2data:
    if index % ratio == 0:
        dev_data.extend(english_instruction2data[i])
    else:
        train_data.extend(english_instruction2data[i])
    index += 1
    
print(len(train_data))
print(len(dev_data))
f = open("/mnt/data/shesj/Data/RL4CoTData/rm_data/{}-onlycorrect-train.json".format(target),'w')
json.dump(train_data,f,indent=2,ensure_ascii=False)
f = open("/mnt/data/shesj/Data/RL4CoTData/rm_data/{}-onlycorrect-dev.json".format(target),'w')
json.dump(dev_data,f,indent=2,ensure_ascii=False)
