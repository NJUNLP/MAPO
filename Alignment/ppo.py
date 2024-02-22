import sys
sys.path.append("/mnt/data/shesj/RL4CoT")
from transformers import AutoTokenizer, GenerationConfig
from dataclasses import dataclass, field
from typing import Optional
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import BitsAndBytesConfig, HfArgumentParser,AutoModelForCausalLM
from peft import PeftModel, PeftConfig,get_peft_model
from transformers import AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, MyPPOTrainer
from trl.core import LengthSampler
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME
from trl import PreTrainedModelWrapper
from utils.generatePrompt import Prompter,get_prompter
from utils.load_model import load_casual_value_head_model
from accelerate import Accelerator
from RM_Function import reward_function_market
VALUE_HEAD_FILE_NAME = "value_head.bin"
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

current_device = Accelerator().local_process_index

def get_state_dict(model: torch.nn.Module, trainable_only: Optional[bool] = True):
    if isinstance(model,AutoModelForCausalLMWithValueHead):
        state_dict = model.pretrained_model.state_dict()
    else:
        state_dict = model.state_dict()
    print("Enter")
    for k, v in state_dict.items():
        print(k)
    filtered_state_dict = {}
    for k, v in model.named_parameters():
        if 'v_head' in k:
            continue
        k = k.replace("pretrained_model.",'')
        print(k)
        filtered_state_dict[k] = state_dict[k].cpu().clone().detach()
    return filtered_state_dict

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Local rank for distributed training (-1: not distributed)"})
    training_config: Optional[str] = field(default=None, metadata={"help": "Path to training config"})
    log_with: Optional[str] = field(default='tensorboard', metadata={"help": "use 'wandb' to log with wandb"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})
    output_dir : Optional[str] = field(default="/mnt/data/shesj/Trained/RL4CoT/")


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
import json
f = open("./ppo_config/" + script_args.training_config,'r')
training_details = json.load(f)
script_args.model_name = training_details['model_name']
script_args.reward_functon = training_details['reward_functon']


prompter = get_prompter(script_args.model_name)

project_name = "{model}-{setting}".format(model=script_args.model_name.split("/")[-1],setting = script_args.training_config)
script_args.output_dir = script_args.output_dir + training_details['version_name'] + '/' + project_name
script_args.explore_data = training_details['explore_data']
reward_function = reward_function_market[script_args.reward_functon]

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name,padding_side = "left")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

map_question2solution = {}

f = open(script_args.explore_data)
data_temp = json.load(f)
for i in data_temp:
    map_question2solution[i["en_question"]] = i["en_collected_answer"]
print(len(map_question2solution))

def tokenize(sample):
    sample['text'] =sample['instruction']
    sample["input_ids"] = tokenizer.encode(sample["text"],max_length=1024,truncation=True,padding='longest')
    sample["query"] = tokenizer.decode(sample["input_ids"])
    sample["resonse_label"] = str(sample["answer"]), 
    #print(sample["en_question"])
    #map_question2solution[sample["en_question"]] = sample["en_collected_answer"]
    return sample


def create_and_prepare_dataset(config):
    ds = load_dataset("json", data_files=config.explore_data)['train'].shuffle(seed=44)

    ds = ds.map(tokenize, batched=False)
    ds = ds.filter(
        lambda x: len(x["input_ids"])  <= 512
    )

    ds.set_format(type="torch")
    return ds



lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules = ["q_proj","v_proj", "o_proj"],
    #target_modules=["gate_proj", "down_proj", "up_proj"]
)


import os
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    
actor_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    device_map={"": current_device},
    torch_dtype=torch.bfloat16,
)
actor_model = get_peft_model(actor_model, lora_config)
    
    
critic_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    device_map={"": current_device},
    torch_dtype=torch.bfloat16,
)
critic_model = get_peft_model(critic_model, lora_config)
critic_model = AutoModelForCausalLMWithValueHead.from_pretrained(critic_model,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={"": current_device})
# critic_model = get_peft_model(critic_model, lora_config)
critic_model.train()


from rlhf_model import RLHFModel
model = RLHFModel(actor_model,critic_model)


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
rm_model_base = AutoModelForSeq2SeqLM.from_pretrained("/mnt/data/shesj/PLM/nllb-200-distilled-600M",device_map={"": current_device})

rm_model_base = rm_model_base.eval()
rm_tokenizer = AutoTokenizer.from_pretrained("/mnt/data/shesj/PLM/nllb-200-distilled-600M")
rm_model = (rm_model_base,rm_tokenizer)

dataset = create_and_prepare_dataset(script_args)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

project_kwargs={"logging_dir": "/mnt/data/shesj/Log/PPO/" + project_name}

config = PPOConfig(
    model_name=script_args.model_name,
    log_with=script_args.log_with,
    learning_rate=training_details['lr'],
    batch_size=training_details['batch_size'],
    mini_batch_size=training_details['mini_batch_size'],
    max_grad_norm=1,
    use_score_scaling=True,
    gradient_accumulation_steps=training_details['gradient_accumulation_steps'],
    ppo_epochs=training_details['ppo_epoch'],
    early_stopping=True,
    optimize_cuda_cache=True,
    seed=script_args.seed,
    project_kwargs = project_kwargs,
    remove_unused_columns=False,
    optimize_device_cache=True
)

from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
policy_model_param_name = []
policy_model_param = []
critic_model_param_name = []
critic_model_param = []
for k,v in model.policy_model.named_parameters():
    if v.requires_grad==True:
        policy_model_param_name.append(k)
        policy_model_param.append(v)

for k,v in model.critic_model.named_parameters():
    if 'lora' in k:
        v.requires_grad = True
    if v.requires_grad == True:
        critic_model_param_name.append(k)
        critic_model_param.append(v)

print("policy_model_para",policy_model_param_name[-5:])
print("critic_model_param",critic_model_param_name[-5:])


optimizer = AdamW([
        {"params":policy_model_param,"lr":config.learning_rate,"eps":1e-5,"betas":(0.9, 0.95),"weight_decay":0.1},
        {"params":critic_model_param,"lr":config.learning_rate * 2,"eps":1e-5,"betas":(0.9, 0.95),"weight_decay":0.1}
        ])


lr_scheduler = get_linear_schedule_with_warmup(optimizer = optimizer,num_warmup_steps = training_details['num_warmup_steps'],num_training_steps=training_details['num_training_steps'])

ppo_trainer = MyPPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer = optimizer,
    lr_scheduler = lr_scheduler,
)


generation_config = GenerationConfig(
    top_p=1.0,
    top_k=0,
    max_new_tokens=512,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


use_pivot = False
dev_pivot = "Tom's ship can travel at 10 miles per hour.  He is sailing from 1 to 4 PM.  He then travels back at a rate of 6 mph.  How long does it take him to get back?"
oracle_output = 5
pivot_input = prompter.generate_prompt(dev_pivot)
pivot_input_id = tokenizer.encode(pivot_input)

    
step = 0
accuracy = 0
total = 0
steps = 0
tokens = 0

torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
)

generated_sample = []

for i in range(300):
    print("EPOCH ",i)
    print(len(ppo_trainer.dataloader))
    for iteration, batch in tqdm(enumerate(ppo_trainer.dataloader)): 
        
        question_tensors = batch["input_ids"]
        if use_pivot:
            question_tensors = [torch.tensor(pivot_input_id).cuda()] + question_tensors

        ppo_trainer.accelerator.unwrap_model(model).policy_model.gradient_checkpointing_disable()
        response_tensors = ppo_trainer.generate(
            question_tensors,
            return_prompt=False,
            generation_config = generation_config,
            #**generation_kwargs,
        )
        ppo_trainer.accelerator.unwrap_model(model).policy_model.gradient_checkpointing_enable()

        batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        labels = [map_question2solution[q] for q in batch['en_question']]
        
        if use_pivot:
            texts = [q + r for q, r in zip([pivot_input] + batch["query"], batch["response"])]
        else:
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]


        status_list = []
        for _,l in zip(texts,labels):
            status_list.append(reward_function(tokenizer,rm_model,_,l))

        rewards = [torch.tensor(_['reward']) for _ in status_list]

        print("================== Step {} ======================".format(step) )   
        print("Lr: " ,ppo_trainer.optimizer.param_groups[0]['lr'])
        print("question: ",batch["query"][0])
        print("label: ",batch["resonse_label"][0])
        print("generate: ",batch["response"][0])
        print("reward: ",rewards[0])
        print(status_list[0])
        

        for _,l,s,labe in zip(texts,labels,status_list,batch["resonse_label"]):
            resonse = _.split('### Response:')[1].strip()
            record_sample = {'step':step,'question':_.split('### Response:')[0].strip(),'answer':resonse,'label':l}
            for k in s:
                record_sample[k] = s[k]
            generated_sample.append(record_sample)
            if "NONE ANSWER" in _:
                continue
            print(resonse)
            print(labe)
            if abs(extract_last_num(resonse) - extract_last_num(labe[0])) <= 1e-2:
                accuracy += 1

            total += 1
            
        with open("/mnt/data/shesj/Log/PPO/" + project_name + "/generated_sample_p{}.json".format(script_args.local_rank), 'w') as f:
            json.dump(generated_sample, f, indent=4)
        
        if use_pivot:
            question_tensors = question_tensors[1:]
            response_tensors = response_tensors[1:]
            rewards = rewards[1:]
        stats = ppo_trainer.step(question_tensors, response_tensors, rewards)

        stats["env/accuracy"] = accuracy / total

        
        step += 1
        torch.cuda.empty_cache()
        ppo_trainer.log_stats(stats, batch, rewards)   
        if step % 30 == 0:
            ppo_trainer.save_pretrained(script_args.output_dir + f"/ModelSaved/step_{step}")
