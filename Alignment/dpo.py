# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional
from peft import get_peft_model, LoraConfig
import torch
from datasets import Dataset, load_dataset,concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments,LlamaForCausalLM
from transformers import TrainerCallback,TrainerState,TrainingArguments,TrainerControl
import shutil
from trl import DPOTrainer
import os
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_dataset, concatenate_datasets, DatasetDict

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        if args.local_rank == 0:
            dir_lst = os.listdir(checkpoint_folder)
            global_step_path=""
            for dir_ele in dir_lst:
                if "global_step" in dir_ele:
                    global_step_path=os.path.join(checkpoint_folder,dir_ele)
            if os.path.exists(global_step_path):
                shutil.rmtree(global_step_path)
        return control



# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    training_config: Optional[str] = field(default=None, metadata={"help": "Path to training config"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    #optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=8, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )


    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=500, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=5, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )

def extract_prompt(prompt_and_response):
    if "### Response: Let's think step by step.\n"  in prompt_and_response:
        search_term = "### Response: Let's think step by step.\n"
    elif "### Response:"  in prompt_and_response:
        search_term = "### Response:"
    else:
        print("No matched")
        print(prompt_and_response)
        exit()
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(args) -> Dataset:
    datasets_list = [load_dataset("json", data_files=i + '-train.json')["train"] for i in args.dataset]
    train_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)

    datasets_list = [load_dataset("json", data_files=i + '-dev.json')["train"] for i in args.dataset]
    dev_dataset = concatenate_datasets(datasets_list).shuffle(seed=42)
    if len(dev_dataset) > 20000:
        dev_dataset = dev_dataset.select(range(20000))

    #train_dataset = load_dataset("json", data_files=args.dataset + '-train.json', split="train").shuffle(seed=42)
    K = args.per_device_train_batch_size * 8 * args.gradient_accumulation_steps * args.max_steps * 2
    print("=============== We are selecting Top ",K, " ======================")
    K = min(K,len(train_dataset))
    train_dataset = train_dataset.select(range(K))
    #dev_dataset = load_dataset("json", data_files=args.dataset + '-dev.json', split="train").shuffle(seed=42)
    
    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_prompt(sample["accept"])
        return {
            "prompt": prompt,
            "chosen": sample["accept"][len(prompt) :],
            "rejected": sample["reject"][len(prompt) :],
        }

    return train_dataset.map(split_prompt_and_responses,num_proc=80),dev_dataset.map(split_prompt_and_responses,num_proc=80)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    import json
    f = open("./dpo_config/" + script_args.training_config,'r')
    training_details = json.load(f)
    script_args.model_name_or_path = training_details['model_name_or_path']
    script_args.dataset = training_details['dataset']
    script_args.beta = training_details['beta']
    script_args.learning_rate = training_details['learning_rate']
    script_args.max_steps = training_details['max_step']
    script_args.gradient_accumulation_steps = training_details['gradient_accumulation_steps']
    script_args.output_dir = "/mnt/data/shesj/Trained/RL4CoT/DPO/" +  script_args.training_config
    

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    # peft_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules = ["q_proj","v_proj","gate_proj", "down_proj", "up_proj"],
    # )
    
    # model = get_peft_model(model, peft_config)
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.bfloat16,
    #     #load_in_4bit=True,
    # )

    model.config.use_cache = False

    #model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path,torch_dtype=torch.bfloat16)

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # model_ref = AutoPeftModelForCausalLM.from_pretrained(
    #     script_args.model_name_or_path,
    #     low_cpu_mem_usage=True,
    #     torch_dtype=torch.bfloat16,
    # )

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    model_ref.config.use_cache = False
    # peft_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    #     target_modules = ["q_proj","v_proj","gate_proj", "down_proj", "up_proj"],
    # )
    
    # model_ref = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.padding_side = "right"  # Allow batched inference

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    train_dataset,eval_dataset = get_hh(script_args)

    # train_dataset = train_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )
    # eval_dataset = eval_dataset.filter(
    #     lambda x: len(x["prompt"]) + len(x["chosen"]) <= script_args.max_length
    #     and len(x["prompt"]) + len(x["rejected"]) <= script_args.max_length
    # )

    # 4. initialize training arguments:


    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_dir="/mnt/data/shesj/Log/DPO/" + script_args.output_dir.split('/')[-1] + '/logs',
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        #optim='adamw_torch',
        bf16=True,
        tf32=True,
        remove_unused_columns=False,
        max_grad_norm= 1,
    )


    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        callbacks=[SavePeftModelCallback]
    )

    # dpo_trainer = DPOTrainer(
    #     model,
    #     model_ref,
    #     args=training_args,
    #     beta=script_args.beta,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     max_length=script_args.max_length,
    #     #max_target_length=script_args.max_target_length,
    #     max_prompt_length=script_args.max_prompt_length,
    #     deepspeed="/mnt/data/sheshuaijie/Code/RL4CoT/PPO/deepspeed_zero2.json",
    #     #deepspeed="ds_config_zero2.json"
    #     #deepspeed_config="/mnt/data/sheshuaijie/Code/RL4CoT/PPO/deepspeed_config.json",
    # )

    # 6. train
    dpo_trainer.train()