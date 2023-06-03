import json
import logging
import os
from typing import Union, List

import torch
from tqdm import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

from generator.base_generator import BaseGenerator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class DollyGenerator(BaseGenerator):
  def __init__(
    self,
    model_params: str,
    base_path: str,
    dataset: str,
    mode: str,
  ) -> None:
    super().__init__()
    self.model_params = model_params
    self.base_path = base_path
    self.mode = mode
    self.model = pipeline(model=f"databricks/dolly-v2-{model_params}",
                          torch_dtype=torch.bfloat16,
                          trust_remote_code=True,
                          device_map="auto",
                          num_return_sequences=1)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, 'llm-generation/', f'dolly-v2-{model_params}/', ), exist_ok=True)
    self.examples = self.read_json(self.process_path_by_dataset(base_path, dataset, mode))

  def _instruction(self, inputs: str = None):
    if inputs is not None:
      return inputs
    return "### Instruction: Given sentence, generate the single explanation sentence that can be the characteristics " \
           "distinguishing the image."

  def build_prompt(self, caption: str) -> Union[str, List[str]]:
    prompt = self._instruction() + "\n"
    prompt += f'Input: {caption}\n'
    prompt += f"### Response: "
    return prompt

  def generate(self) -> Union[str, List[str]]:

    with open(os.path.join(self.base_path, 'llm-generation', f'dolly-v2-{self.model_params}/semantic_unit.{self.mode}.json'),
              'a+') as f:
      for example in tqdm(self.examples, total=len(self.examples), desc=f'generating examples '
                                                                        f'with databricks/dolly-v2-{self.model_params}'):
        caption = example['caption']
        prompt = self.build_prompt(caption)
        generated_texts = self.model(prompt)
        obj = json.dumps({
          'prompt': prompt,
          'semantic_units': generated_texts
        }, ensure_ascii=False)
        f.write(obj + '\n')


if __name__ == '__main__':
  generator = DollyGenerator(model_params='12b', base_path='../rsc/', dataset='cirr', mode='train')
  generator.generate()

  pass
