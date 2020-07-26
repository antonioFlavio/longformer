import logging
import os
import gc
from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser, BertForMaskedLM, BertTokenizerFast
from util import Util
# Temporário 
#from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import AutoModel, AutoTokenizer
from transformers import __version__ as teste

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -- Set local folder root directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# ----------------

@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
    max_pos: int = field(default=1024, metadata={"help": "Maximum position"})

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
    '--gradient_accumulation_steps', '32',
    '--evaluate_during_training',
    '--do_train',
    '--do_eval'
])

training_args.val_datapath = 'wikiportuguese/wiki.test.raw'
training_args.train_datapath = 'wikiportuguese/wiki.train.raw'

#model_name = "bert-base-multilingual-cased"
model_name = "neuralmind/bert-base-portuguese-cased"
model_path = f'{training_args.output_dir}/{model_name}-{model_args.max_pos}'

if not os.path.exists(model_path):
    os.makedirs(model_path)

# model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
# Util.pretrain_and_evaluate(training_args, model, tokenizer, eval_only=True, model_path=None)

logger.info(f'Converting roberta-base into roberta-base-{model_args.max_pos}')
model, tokenizer = Util.create_long_model(model_name=model_name,save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)
Util.pretrain_and_evaluate(training_args, model, tokenizer, eval_only=False, model_path=None)
