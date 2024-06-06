
import random
import json
random.seed(42)
import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose","template_name")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        self.template_name = template_name
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        print("Using prompt template: ",file_name)
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input_context: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input_context:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input_context
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        if self.template["response_split"] not in output:
            print(output)
            exit()
        return output.split(self.template["response_split"])[1].strip()

import re


def get_prompter(model_path):
    if 'parallel' in model_path.lower():
        prompt_template_name = 'parallel'
        print("Using parallel template")
    else:
        print("Error with not found template")
        prompt_template_name = 'parallel'
    prompt = Prompter(template_name=prompt_template_name)
    return prompt