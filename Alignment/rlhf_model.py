import torch.nn as nn
import torch


def make_model_gradient_checkpointing_compatible(model):
    # Higgingface added this enable input require grads function to make gradient checkpointing work for lora-only optimization
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "get_input_embeddings"):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad)
    return model

class RLHFModel(nn.Module):
    def __init__(self, policy_model, critic_model) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.policy_model.gradient_checkpointing_enable()
        self.policy_model = make_model_gradient_checkpointing_compatible(
                    self.policy_model)
        self.critic_model = critic_model
        self.critic_model.gradient_checkpointing_enable()
        self.critic_model = make_model_gradient_checkpointing_compatible(
                    self.critic_model)
    
    def train(self,mode=True):
        self.policy_model.train(mode)
        self.critic_model.train(mode)
        
    def eval(self):
        self.policy_model.eval()
        self.critic_model.eval()

    def forward(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs):
        return self.forward_policy(input_ids,past_key_values,attention_mask,**kwargs),self.forward_critic(input_ids,past_key_values,attention_mask,**kwargs)

    def forward_policy(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        base_model_output = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        # force upcast in fp32 if logits are in half-precision
        if lm_logits.dtype != torch.float32:
            lm_logits = lm_logits.float()
        return lm_logits

    def forward_critic(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the logits of the value head.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        """
        kwargs["output_hidden_states"] = True  # this had already been set in the LORA / PEFT examples
        _, _, value = self.critic_model(input_ids=input_ids,past_key_values=past_key_values,attention_mask=attention_mask,**kwargs)

        return value

    # def generate(self, *args, **kwargs):
    #     r"""
    #     A simple wrapper around the `generate` method of the wrapped model.
    #     Please refer to the [`generate`](https://huggingface.co/docs/transformers/internal/generation_utils)
    #     method of the wrapped model for more information about the supported arguments.

    #     Args:
    #         *args (`list`, *optional*):
    #             Positional arguments passed to the `generate` method of the wrapped model.
    #         **kwargs (`dict`, *optional*):
    #             Keyword arguments passed to the `generate` method of the wrapped model.
    #     """
    #     return self.pretrained_model.generate(*args, **kwargs)

    # def state_dict(self, *args, **kwargs):
    #     r"""
    #     Returns the state dictionary of the model. We add the state dictionary of the value head
    #     to the state dictionary of the wrapped model by prepending the key with `v_head.`.
    #     """
    #     if not self.is_peft_model:
    #         pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
    #     else:
    #         # if it is a peft model, only save the v_head
    #         pretrained_model_state_dict = {}

    #     v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
    #     for k, v in v_head_state_dict.items():
    #         pretrained_model_state_dict[f"v_head.{k}"] = v
    #     return pretrained_model_state_dict