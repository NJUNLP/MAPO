from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, BitsAndBytesConfig
from transformers import AutoModelForSequenceClassification,AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from datasets import *
import os
import numpy as np
import torch
import json
from transformers.trainer_pt_utils import nested_detach
from peft import PeftModel, PeftConfig,get_peft_model,LoraConfig
import tempfile
VALUE_HEAD_FILE_NAME = "value_head.bin"
def load_valuehead_params(model: torch.nn.Module, checkpoint_dir: os.PathLike) -> bool:
    valuehead_file = os.path.join(checkpoint_dir, VALUE_HEAD_FILE_NAME)
    valuehead_state_dict = torch.load(valuehead_file, map_location="cpu")
    model.register_buffer("reward_head_weight", valuehead_state_dict["summary.weight"])
    model.register_buffer("reward_head_bias", valuehead_state_dict["summary.bias"])
    model.register_buffer("default_head_weight", torch.zeros_like(valuehead_state_dict["summary.weight"]))
    model.register_buffer("default_head_bias", torch.zeros_like(valuehead_state_dict["summary.bias"]))
    return True

def load_casual_value_head_model(sft_path,lora_path,device_map="auto"):
    #tokenizer = AutoTokenizer.from_pretrained(sft_path,padding_side = "left")
    tokenizer = AutoTokenizer.from_pretrained(sft_path)
    #tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            load_in_8bit=False,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True
        ) 
    model = PeftModel.from_pretrained(model, lora_path)
    model =  AutoModelForCausalLMWithValueHead.from_pretrained(model)
    if load_valuehead_params(model, lora_path):
        model.v_head.load_state_dict({
            "summary.weight": getattr(model, "reward_head_weight"),
            "summary.bias": getattr(model, "reward_head_bias")
        })
    model.eval()
    return tokenizer,model