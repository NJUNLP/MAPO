import os
import json
import argparse
# 指定输入文件夹和输出文件

parser = argparse.ArgumentParser(description='merging')
parser.add_argument('--source_dir', type=str)
parser.add_argument('--target_file', type=str)

args = parser.parse_args()

input_folder =  args.source_dir # 更改为您的输入文件夹路径
output_file = args.target_file   # 更改为输出文件名

# 初始化一个空的列表来存储合并的JSON数据
merged_data = []

# 遍历输入文件夹中的所有JSON文件
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                merged_data = merged_data + data
            except json.JSONDecodeError as e:
                print(f"无法解析文件 {filename}: {e}")


print(len(merged_data))
def test_unique(data):
    quesiton2data = {}
    hit = 0
    for i in data:
        if 'question' not in i:
            i['question'] = i['prompt']
        if i['question'] in quesiton2data:
            #print("has same question")
            hit += 1
        else:
            quesiton2data[i['question']] = 1
    print(hit)
    
test_unique(merged_data)
print(len(merged_data))
# 合并的JSON数据写入输出文件
with open(output_file, 'w') as merged_json_file:
    json.dump(merged_data, merged_json_file, indent=4,ensure_ascii=False)

print(f"已将所有JSON文件合并到 {output_file}")
