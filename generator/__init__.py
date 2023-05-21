import logging

from generator.dolly_generator import DollyGenerator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


GENERATOR = {
  'dolly': DollyGenerator,
  'gpt-j': None,
}