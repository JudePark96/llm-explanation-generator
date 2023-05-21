import logging
import os
from typing import Union, List

import torch
from tqdm import tqdm
from transformers import pipeline

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
    # model_params => 3b, 7b, 12b
    self.model_params = model_params
    self.model = pipeline(model=f"databricks/dolly-v2-{model_params}",
                          torch_dtype=torch.bfloat16,
                          trust_remote_code=True,
                          device_map="auto")

    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, f'dolly-v2-{model_params}/', ), exist_ok=True)
    self.examples = self.read_json(self.process_path_by_dataset(base_path, dataset, mode))

  def _instruction(self):
    return "Instruction: Given a context, extract the phrases that can be the characteristics distinguishing ' \
           'the image such as keypoint, entity, color, and shape with explanation."


  def build_prompt(self, caption: str) -> Union[str, List[str]]:
    prompt = self._instruction() + "\n"
    prompt += f'Context: {caption}'
    prompt += "\n"
    return prompt

  def generate(self) -> Union[str, List[str]]:
    for example in tqdm(self.examples, total=len(self.examples), desc=f'generating examples '
                                                                      f'with databricks/dolly-v2-{self.model_params}'):
      caption = example['caption']
      prompt = self.build_prompt(caption)
      a = self.model(prompt)
      from pprint import pprint
      pprint(a[0]['generated_text'])
    pass


if __name__ == '__main__':
  """
  ('Given a sentence, extract the phrases that can be the characteristics '
 'distinguishing the image such as keypoint, entity, color, and shape. Extract '
 'as many phrases as possible.\n'
 'Table and chairs turn to more dark color and has a gray carpet.\n'
 'Turn to more dark color and has a gray carpet.')
  """

  generator = DollyGenerator(model_params='12b', base_path='../rsc/', dataset='cirr', mode='train')
  generator.generate()

  pass
