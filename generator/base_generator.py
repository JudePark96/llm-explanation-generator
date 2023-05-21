import abc
import json
import logging
import os.path
from abc import abstractmethod
from typing import Union, List, Dict, Any

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

base_instruction = 'Instruction: Given a sentence, extract the phrases that can be the characteristics distinguishing ' \
                   'the image such as keypoint, entity, color, and shape. Extract as many phrases as possible. ' \
                   'The answer should be json format having key as {"phrase_1": "description of this phrase", ' \
                   '"phrase_2": "description of this phrase", ...}. ' \
                   'And give me an answer without explanation.'


class BaseGenerator(metaclass=abc.ABCMeta):
  def process_path_by_dataset(self, base_path: str, dataset: str, mode: str) -> str:
    if dataset == 'cirr':
      file_name = f'cap.rc2.{mode}.json'
      path = os.path.join(base_path, f'dataset/{dataset}', file_name)
      return path
    elif dataset == 'fashioniq':
      pass

  def read_json(self, inputs: str) -> Union[List[Dict[str, str]], Dict[str, str]]:
    with open(inputs, 'rb') as f:
      examples = json.load(f)

    return examples

  def save_json(self, obj: Any, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(obj, f, ensure_ascii=False, indent=2)

  @abstractmethod
  def build_prompt(self, caption: str) -> Union[str, List[str]]:
    raise NotImplementedError()

  @abstractmethod
  def generate(self) -> Union[str, List[str]]:
    raise NotImplementedError()
