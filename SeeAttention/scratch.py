
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

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)

outputs[-1][5][0][11] ## last entry. return 12 heads of the last layer.  from A --> to --> B
torch.sum(outputs[-1][5][0][11],1) ## row sum to 1. 
## so 1 row, sums to 1, this means that . for row i, we see how much col j contributes to it

prediction_scores, seq_relationship_scores = outputs[:2]


