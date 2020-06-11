# This can be used to not pretrained and pretrained long models.

import sys
import logging
from transformers import RobertaTokenizerFast
from .long_models import RobertaLongForMaskedLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model_path = sys.argv[1]

logger.info(f'Loading the model from {model_path}')
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaLongForMaskedLM.from_pretrained(model_path)