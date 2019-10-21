
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


word_dot_distance = torch.randn(2,4,3) ## 2 batch
word_word_relation = torch.LongTensor ( np.round ( np.random.uniform(size=(2,4,4),low=0,high=2) ) ) 
out = torch.gather( word_dot_distance, dim=2, index=word_word_relation )

