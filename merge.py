import json
import logging
import os.path
import sys

from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  if sys.argv[1] == 'dolly':
    assert sys.argv[2] in ['3b', '7b', '12b']
    assert sys.argv[3] in ['train', 'val', 'test1']

    base_dir = os.path.join(f'./rsc/llm-generation/dolly-v2-{sys.argv[2]}/')

    # semantic_unit.train.json
    with open(os.path.join(f'./rsc/dataset/cirr/cap.rc2.{sys.argv[3]}.json'), 'rb') as f:
      dataset = json.load(f)

    with open(os.path.join(base_dir, f'semantic_unit.{sys.argv[3]}.json'), 'r', encoding='utf-8') as f:
      lm_predictions = [json.loads(i) for i in f]

    assert len(dataset) == len(lm_predictions), '%d v.s. %d' % (len(dataset), len(lm_predictions))

    for example, prediction in tqdm(zip(dataset, lm_predictions), total=len(dataset),
                                    desc='merging existing dataset and lm predictions.'):
      explanation: str = prediction['semantic_units'][0]['generated_text'].split('### Response:')[-1].strip()
      explanation = ' '.join([i.strip() for i in explanation.split('\n')])
      example['semantic_units'] = explanation

    target_dir = os.path.join(base_dir, 'semantic_units')
    os.makedirs(target_dir, exist_ok=True)

    with open(os.path.join(target_dir, f'semantic.cap.rc2.{sys.argv[3]}.json'), 'w', encoding='utf-8') as f:
      json.dump(dataset, f, ensure_ascii=False, indent=2)