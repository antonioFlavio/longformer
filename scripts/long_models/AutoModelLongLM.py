from transformers import AutoModel, AutoTokenizer
from transformers.modeling_longformer import LongformerSelfAttention

class AutoModelLongLM(AutoModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)
    
    def from_pretrained(self, model_path):
        super().from_pretrained(model_path)