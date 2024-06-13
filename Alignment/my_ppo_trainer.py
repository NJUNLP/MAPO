'''
Author: Shuaijie She

This code is a modified version of the original trl ppo_trainer, attempting to incorporate two features to enhance the stability and effectiveness of the training process:
1. During the early stages of training, the policy model is frozen, and the critic model is warmed up.
2. The policy model and critic model do not share parameters; instead, they are trained using independent parameters.

Given that the official trl library has also introduced many new features, it is recommended to use the latest official implementation from trl for training.
https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
https://github.com/huggingface/trl/blob/main/trl/trainer/ppov2_trainer.py

If you still wish to use this my modified version of the implementation, you will need to place this code in the source of the trl library that you have installed via pip. Additionally, you must add this class to the corresponding __init__.py and other modules to ensure that the module can be found and executed correctly when needed.
'''


from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import gc
import random
import warnings
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import top_k_top_p_filtering
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)



if is_deepspeed_available():
    import deepspeed


class PPODecorators(object):
    optimize_device_cache = False

    @classmethod
    @contextmanager
    def empty_device_cache(cls):
        yield
        if cls.optimize_device_cache and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/huggingface/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class MyPPOTrainer(PPOTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
            details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`PreTrainedTokenizerBase`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model = None,
        ref_model = None,
        tokenizer = None,
        dataset = None,
        optimizer = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        critic_warmup_setps: Optional[int] = 0,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                Hugging Face tokenizer
            dataset (Optional[Union[`torch.utils.data.Dataset`, `datasets.Dataset`]]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config ,model,ref_model,tokenizer,dataset,optimizer,data_collator,num_shared_layers,lr_scheduler)
        # initial seed for reproducible experiments
        self.critic_warmup_setps = critic_warmup_setps
        if self.critic_warmup_setps != 0:
            self.requires_grad_param_name = []
            for k,v in self.model.named_parameters():
                if v.requires_grad==True:
                    self.requires_grad_param_name.append(k)
            print("================== Setting Only VHead Update ==================")
            for k,v in model.named_parameters():
                if 'v_head' not in k:
                    v.requires_grad=False#固定参数
         



    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        
        if self.critic_warmup_setps > 0:
            print("========================= critic_warmup_setps {} Steps left =========================".format(self.critic_warmup_setps))
            loss = loss_v
            self.critic_warmup_setps -= 1
        else:
            loss = loss_p + loss_v
            
        if self.critic_warmup_setps == 1:
            print("================== Reopen all Para Update ==================")
            self.model_params
            for k,v in self.model.named_parameters():
                if k in self.requires_grad_param_name:
                    v.requires_grad=True

        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model_params, self.config.max_grad_norm)
        self.optimizer.step()
        # we call optimizer.zero_grad() every time and let `accelerator` handle accumulation
        # see https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#the-finished-code
        self.optimizer.zero_grad()
        return train_stats
