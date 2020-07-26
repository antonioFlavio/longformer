import logging
import math
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, HfArgumentParser
from transformers.modeling_longformer import LongformerSelfAttention
# Temporário
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from transformers import BertForMaskedLM, BertTokenizerFast, AutoModelWithLMHead
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Util:
    @staticmethod
    def create_long_model(model_name, save_model_to, attention_window, max_pos):        
        model = BertForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_pos)
        config = model.config

        # extend position embeddings
        tokenizer.model_max_length = max_pos
        tokenizer.init_kwargs['model_max_length'] = max_pos        
        current_max_pos, embed_size = model.bert.embeddings.position_embeddings.weight.shape
        
        config.max_position_embeddings = max_pos
        assert max_pos > current_max_pos
        # allocate a larger position embedding matrix        
        new_pos_embed = model.bert.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
        # copy position embeddings over and over to initialize the new position embeddings
        k = 0
        step = current_max_pos

        while k < max_pos - 1:            
            if(k + step < max_pos):
                new_pos_embed[k:(k + step)] = model.bert.embeddings.position_embeddings.weight
            k += step
        
        model.bert.embeddings.position_embeddings.weight.data = new_pos_embed

        # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
        config.attention_window = [attention_window] * config.num_hidden_layers
        
        for i, layer in enumerate(model.bert.encoder.layer):
            longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
            longformer_self_attn.query = layer.attention.self.query
            longformer_self_attn.key = layer.attention.self.key
            longformer_self_attn.value = layer.attention.self.value

            longformer_self_attn.query_global = layer.attention.self.query
            longformer_self_attn.key_global = layer.attention.self.key
            longformer_self_attn.value_global = layer.attention.self.value

            layer.attention.self = longformer_self_attn

        logger.info(f'saving model to {save_model_to}')
        model.save_pretrained(save_model_to)
        tokenizer.save_pretrained(save_model_to)
        return model, tokenizer    
    
    @staticmethod
    def copy_proj_layers(model):        
        for i, layer in enumerate(model.bert.encoder.layer):
            layer.attention.self.query_global = layer.attention.self.query
            layer.attention.self.key_global = layer.attention.self.key
            layer.attention.self.value_global = layer.attention.self.value
        return model

    @staticmethod
    def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path=None, block_size=''):
        val_dataset = TextDataset(tokenizer=tokenizer,
                              file_path=args.val_datapath,
                              block_size=tokenizer.max_len)
        if eval_only:
            train_dataset = val_dataset
        else:
            logger.info(f'Loading and tokenizing training data is usually slow: {args.train_datapath}')
            train_dataset = TextDataset(tokenizer=tokenizer,
                                        file_path=args.train_datapath,
                                        block_size=tokenizer.max_len)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        trainer = Trainer(model=model, args=args, data_collator=data_collator,
                        train_dataset=train_dataset, eval_dataset=val_dataset, prediction_loss_only=True,)

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        logger.info(f'Initial eval bpc: {eval_loss/math.log(2)}')
        
        if not eval_only:
            trainer.train(model_path=model_path)
            trainer.save_model()

            eval_loss = trainer.evaluate()
            eval_loss = eval_loss['eval_loss']
            logger.info(f'Eval bpc after pretraining: {eval_loss/math.log(2)}')