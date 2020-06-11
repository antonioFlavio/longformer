import logging
import os
import sys
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser
from util import Util
# Temporário
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_longformer import LongformerSelfAttention

from long_models import AutoModelLongLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -- Set local folder root directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# ----------------

model_path = ''
if len(sys.argv) > 1:
    model_path = sys.argv[1]

if model_path == '':
    model_path = 'tmp\\bert-br-4098'

@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    #'script.py',
    '--output_dir', 'tmp',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_gpu_eval_batch_size', '2',
    '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
    #'--device', 'cuda0',  # one GPU
    '--gradient_accumulation_steps', '2',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval'
])

training_args.val_datapath = 'wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = 'wikitext-103-raw/wiki.train.raw'

path_exists = os.path.exists(model_path)

logger.info(f'Loading the model from {model_path}')

model, tokenizer = Util.create_long_model(save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

# model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
# tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', model_max_length=model_args.max_pos)

logger.info(f'Pretraining roberta-base-{model_args.max_pos} ... ')

Util.pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False)

logger.info(f'Copying local projection layers into global projection layers ... ')
model = Util.copy_proj_layers(model)
logger.info(f'Saving model to {model_path}')
model.save_pretrained(model_path)