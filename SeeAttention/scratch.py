
import torch

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                  BertConfig, BertForMaskedLM, BertTokenizer, BertForPreTraining,
                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)


## extract last layer attention ?? 


config = BertConfig.from_pretrained('bert-base-uncased')
config.output_attentions=True 
config.output_hidden_states=True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForPreTraining(config)
model.eval() 

input_ids1 = tokenizer.encode("Hello, my dog is cute")  # Batch size 1
input_ids2 = tokenizer.encode("Hello, my dog is one")
input_ids = torch.tensor ( [input_ids1,input_ids2] ) 
outputs = model(input_ids)

layers = outputs[-1] ## 12 layers

head = layers[4] 
head.shape ## each layer has 12 heads, head will have num_batch x num_head x word x word 

## last entry. return 12 heads of the last layer.  from A --> to --> B
torch.sum(head[0][11],1) ## row sum to 1. 
## so 1 row, sums to 1, this means that . for row i, we see how much col j contributes to it

prediction_scores, seq_relationship_scores = outputs[:2]


