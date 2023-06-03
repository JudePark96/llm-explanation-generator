import argparse
import logging

from generator import DollyGenerator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_options():
  opts = argparse.ArgumentParser()
  opts.add_argument('--rsc_path', type=str, default='./rsc/')
  opts.add_argument('--llm', type=str, default='dolly')
  opts.add_argument('--params', type=str, default='12b')
  opts.add_argument('--dataset', type=str, default='cirr')
  opts.add_argument('--mode', type=str, default='train')

  return opts.parse_args()


if __name__ == '__main__':
  opts = get_options()
  rsc_path, llm, params, dataset, mode = opts.rsc_path, opts.llm, opts.params, opts.dataset, opts.mode

  assert mode in ['train', 'val', 'test1']
  assert dataset in ['cirr']

  if llm == 'dolly':
    assert params in ['3b', '6b', '7b', '12b'], f"Dolly uses one of '['3b', '6b', '7b', '12b']', not {params}"
    module = DollyGenerator(params, base_path=rsc_path, dataset=dataset, mode=mode)
    module.generate()


    pass
