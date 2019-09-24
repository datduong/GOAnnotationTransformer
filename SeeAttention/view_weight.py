

from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import pickle
import random
import re,sys
import pandas as pd
import numpy as np

import torch
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                  BertConfig, BertTokenizer)

sys.path.append("/local/datdb/BertGOAnnotation")
import KmerModel.TokenClassifier as TokenClassifier
import finetune.evaluation_metric as evaluation_metric


class TextDataset(Dataset):
  def __init__(self, tokenizer, label_2test_array, file_path='train', block_size=512):
    assert os.path.isfile(file_path)
    directory, filename = os.path.split(file_path)
    cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')

    label_2test_array = sorted(label_2test_array) ## just to be sure we keep alphabet
    num_label = len(label_2test_array)
    print ('num label {}'.format(num_label))
    ## faster look up
    label_index_map = { name : index for index,name in enumerate(label_2test_array) }
    label_string = " ".join(label_2test_array)

    if os.path.exists(cached_features_file+'examples'): ## take 1 thing to test if it exists
      logger.info("Loading features from cached file %s", cached_features_file)
      with open(cached_features_file+'examples', 'rb') as handle:
        self.examples = pickle.load(handle)
      with open(cached_features_file+'attention_mask', 'rb') as handle:
        self.attention_mask = pickle.load(handle)
      with open(cached_features_file+'label1hot', 'rb') as handle:
        self.label1hot = pickle.load(handle)
      with open(cached_features_file+'label_mask', 'rb') as handle:
        self.label_mask = pickle.load(handle)
      with open(cached_features_file+'token_type', 'rb') as handle:
        self.token_type = pickle.load(handle)

    else:
      logger.info("Creating features from dataset file at %s", directory)

      self.attention_mask = []
      self.examples = []
      self.label1hot = [] ## take 1 hot (should short labels by alphabet)
      self.label_mask = []
      self.token_type = []

      fin = open(file_path,"r",encoding='utf-8')
      for counter, text in tqdm(enumerate(fin)):
        # if counter > 100 :
        #   break
        text = text.strip()
        if len(text) == 0: ## skip blank ??
          continue

        ## split at \t ?? [seq \t label]
        text = text.split("\t") ## position 0 is kmer sequence, position 1 is list of labels
        kmer_text = text[0].split() ## !! we must not use string text, otherwise, we will get wrong len
        num_kmer_text = len(kmer_text)
        this_label = text[1].split() ## by space

        ## convert label into 1-hot style
        label1hot = np.zeros(num_label) ## 1D array
        index_as1 = [label_index_map[label] for label in this_label]
        label1hot [ index_as1 ] = 1
        self.label1hot.append( label1hot )

        ## create token_type
        token_type = [0]*(num_kmer_text+2) + [1]*(num_label+1) ## add 2 because of cls and sep, ## add 1 because of sep
        WHERE_KMER_END = num_kmer_text + 2 ## add 2 because of sep cls , so we can mask out Kmer, because we will only predict the labels

        ## combine the labels back with the kmer text ??
        ## want kmer + ALL LABELS
        ## ok to use @text[0] because we will tokenize, so we will remove the space
        text = text[0] + " [SEP] " + label_string ## add all the labels
        text_split = tokenizer.tokenize(text)
        tokenized_text = tokenizer.convert_tokens_to_ids(text_split) ## the tab won't matter here ## split at \t ?? [seq \t label]

        if len(tokenized_text) < block_size : ## short enough, so just use it
          ## add padding to match block_size
          tokens = tokenizer.add_special_tokens_single_sentence(tokenized_text) ## [CLS] something [SEP]
          attention_indicator = [1]*len(tokens) + [0]*(block_size-len(tokens))
          tokens = tokens + [0]*(block_size-len(tokens)) ## add padding 0

          assert len(tokens) == block_size
          assert len(attention_indicator) == block_size

          self.examples.append(tokens)
          self.attention_mask.append(attention_indicator)

          token_type = token_type + [0]* (block_size-len(token_type)) ## zero here, because we use mask anyway, so doesn't matter ... add 0-padding to the token types
          self.token_type.append(token_type)

          ## get label_mask, so we test only on labels we want, and not some random tokens
          label_mask = np.zeros( block_size ) ## 0--> not attend to
          label_mask [ WHERE_KMER_END:(WHERE_KMER_END+num_label) ] = 1 ## set to 1 so we can pull these out later, ALL LABELS WILL NEED 1, NOT JUST THE TRUE LABEL
          self.label_mask.append(label_mask)

          if counter < 3:
            print ('\nsee input text\n {}'.format(tokens))

        else:
          print ( 'too long, code unable to split long sentence ... infact we should not split ... block {} len {}'.format(block_size,len(tokenized_text)) )
          exit()
          # while len(tokenized_text) >= block_size:  # Truncate in block of block_size
          #   self.examples.append(tokenizer.add_special_tokens_single_sentence(tokenized_text[:block_size]))
          #   tokenized_text = tokenized_text[block_size:]
          #   attention_indicator = [1]*block_size
          #   self.attention_mask.append(attention_indicator)

      ## save at end
      logger.info("To save read/write time... Saving features into cached file %s", cached_features_file)
      with open(cached_features_file+'examples', 'wb') as handle:
        pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'attention_mask', 'wb') as handle:
        pickle.dump(self.attention_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'label1hot', 'wb') as handle:
        pickle.dump(self.label1hot, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'label_mask', 'wb') as handle:
        pickle.dump(self.label_mask, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'token_type', 'wb') as handle:
        pickle.dump(self.token_type, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, item):
    return (torch.tensor(self.attention_mask[item]), torch.tensor(self.examples[item]),
            torch.LongTensor(self.label1hot[item]), torch.tensor(self.label_mask[item]),
            torch.tensor(self.token_type[item]) )


def load_and_cache_examples(args, tokenizer, label_2test_array, evaluate=False):
  dataset = TextDataset(tokenizer, label_2test_array, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
  return dataset



## does weight make sense ? 

config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
config.output_attentions=True ## override @config
config.output_hidden_states=True


tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

model = TokenClassifier.BertForTokenClassification1hot.from_pretrained(args.model_name_or_path, config=config) ## use @config=config to override the default @config

# model.to(args.device) ## ?? do we need to send to gpu 

## create dataset again.
## must respect the ordering of GO terms. 

# read in labels to be testing
label_2test_array = pd.read_csv(args.label_2test,header=None)
label_2test_array = sorted(list( label_2test_array[0] ))
label_2test_array = [re.sub(":","",lab) for lab in label_2test_array] ## splitting has problem with the ":"

eval_dataset = load_and_cache_examples(args, tokenizer, label_2test_array, evaluate=True)
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
# Note that DistributedSampler samples randomly
eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)


eval_loss = 0.0
nb_eval_steps = 0
model.eval()


for batch in tqdm(eval_dataloader, desc="Evaluating"):
  # batch = batch.to(args.device)

  max_len_in_batch = int( torch.max ( torch.sum(batch[0],1) ) ) ## only need max len
  attention_mask = batch[0][:,0:max_len_in_batch].to(args.device)
  inputs = batch[1][:,0:max_len_in_batch].to(args.device)
  labels = batch[2].to(args.device) ## already in batch_size x num_label
  labels_mask = batch[3][:,0:max_len_in_batch].to(args.device) ## extract out labels from the array input... probably doesn't need this to be in GPU
  token_type = batch[4][:,0:max_len_in_batch].to(args.device)

  with torch.no_grad():
    outputs = model(inputs, token_type_ids=token_type, attention_mask=attention_mask, labels=labels, position_ids=None, attention_mask_label=labels_mask )
    lm_loss = outputs[0]
    eval_loss += lm_loss.mean().item()

  nb_eval_steps += 1
 
  last_layer_att = outputs[-1] ## batch x last_layer_att[layer][head] dim = word x word

  ## because of batch size ... different sequence has different len. how do we align them ?? 
