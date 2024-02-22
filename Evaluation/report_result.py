base_path = "./Generations/"

test_set = [
    "MSVAMP_test",
    "mgsm8k_test",
    "NumGLUE_test_multi",
]

import json
import re
import re

# Examples data
test_setting = [
    "MetaMath-7B-V1.0-{}_genen_collect.json"
]

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0



lang_list = ["Bengali" ,  "Thai", "Swahili", "Japanese",  "Chinese", "Russian",   "German",  "Spanish",   "French"]


no_english_list = lang_list
lang_list  = lang_list + ["English"]
for g in test_set:
    print(g)
    files = [t.format(g) for t in test_setting]
    print('\t'.join(lang_list))
    for path in files:
        try:
            f = open(base_path + path)
        except:
            print(base_path + path, "Not Exist")
            continue
        lang_dict = {}
        data = json.load(f)

        count = 0

        for i in data:
            i['answer'] = str(i['answer'])
            label = extract_last_num(i['answer'])
            if 'instruction' not in i:
                i['instruction'] = i['prompt']
            lang = i['instruction'].split(".\n\n### Instruction:\n")[0].split("Please answer in ")[1]

            if lang not in lang_dict:
                lang_dict[lang] = []

            if 'answers' not in i and 'result' in i:
                i['answers'] = []
                i['answers'].append({'generated':i['result']})
            if abs(extract_last_num(i['answers'][0]['generated']) - label)<1e-2:
                lang_dict[lang].append(1)
            else:
                lang_dict[lang].append(0)
            count += 1

        
        all_avg = 0
        nonEn_avg = 0
        for l in lang_list:
            all_avg += sum(lang_dict[l])* 100/len(lang_dict[l])
            if l != "English":
                nonEn_avg += sum(lang_dict[l])* 100/len(lang_dict[l])
            print(round(sum(lang_dict[l])* 100 /len(lang_dict[l]),1),end='	')
        
        print(round(all_avg/10,1),end='	')
        print(round(nonEn_avg/9,1),end='	')
        print(path.split("/")[-1],end="\t")
        print('')



    print('\n\n\n')







