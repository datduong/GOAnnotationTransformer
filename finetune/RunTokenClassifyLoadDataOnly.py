
## do this on hoffman to save CPU time

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

from scipy.sparse import coo_matrix

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                  BertConfig, BertForMaskedLM, BertTokenizer)

from pytorch_transformers.modeling_bert import BertForPreTraining

logger = logging.getLogger(__name__)

sys.path.append("/u/scratch/d/datduong/BertGOAnnotation")
import KmerModel.TokenClassifier as TokenClassifier
import evaluation_metric

import PosthocCorrect 

MODEL_CLASSES = {
  'bert': (BertConfig, TokenClassifier.BertForTokenClassification2EmbPPI, BertTokenizer) ## replace the standard @BertForTokenClassification
}

def ReadProtData(string,num_aa,max_num_aa,annot_data,annot_name_sorted,evaluate):

  ## must do padding so all items get the same size
  out = np.zeros((max_num_aa,len(annot_name_sorted))) ## maximum possible
  if string == 'none':
    return coo_matrix(out)

  # @string is some protein data, delim by ";"
  annot = string.split(';')
  annot_matrix = np.zeros((num_aa,len(annot))) ## exact @num_aa without CLS and SEP

  for index, a in enumerate(annot): ## we do not need the whole matrix. we can use 1,2,3,4,5 indexing style on the column entry
    ## want annotation on protein sequence into matrix. Len x Type
    ## a = 'COILED 87-172;DOMAIN uba 2-42;DOMAIN ubx 211-293'.split(';')
    a = a.split() ## to get the position, which should always be at the last part
    type_name = " ".join( a[0 : (len(a)-1)] )

    if type_name not in annot_name_sorted: ## unseen Domain Type
      type_number = 1 ## set to UNK
    else:
      ## !! notice, shift by +2 so that we PAD=0 (nothing) and UNK=1 (some unseen domain)
      type_number = annot_name_sorted[ type_name ] ## make sure we exclude position which is last. @annot_name_sorted is index-lookup

    ## in preprocessing, we have -1, because uniprot give raw number, but python starts at 0.
    ## we do not -1 for the end point.
    row = [int(f) for f in a[-1].split('-')] ## get back 2 numbers
    row = np.arange(row[0]-1,row[1]) ## continuous segment

    ## we have to randomly assign UNK... assign a whole block of UNK
    ## do not need to assign random UNK for dev or test set
    if (not evaluate) and (type_number > 1): ## chance of being UNK
      if np.random.uniform() < annot_data[type_name][1]*1.0/annot_data[type_name][0]:
        annot_matrix [ row,index ] = 1 ## type=1 is UNK
      else:
        annot_matrix [ row,index ] = type_number ## @col is the number of the type w.r.t. the whole set of types, @index is just for this string
    else:
      annot_matrix [ row,index ] = type_number

  ## out is max_len (both aa + CSL SEP PAD) + len_annot
  ## read by row, so CLS has annot=0
  ## only need to shift 1 row down
  out[1:(num_aa+1), 0:len(annot)] = annot_matrix ## notice shifting by because of CLS and SEP

  # print ('\nsee annot matrix\n')
  # print (annot_matrix)
  return coo_matrix(out)


class TextDataset(Dataset):
  def __init__(self, tokenizer, label_2test_array, file_path='train', block_size=512, max_aa_len=1024, args=None, evaluate=None):
    # @max_aa_len is already cap at 1000 in deepgo, Facebook cap at 1024

    self.args = args

    assert os.path.isfile(file_path)
    directory, filename = os.path.split(file_path)
    if args.aa_type_emb:
      directory = os.path.join(directory , 'aa_ppi_annot_cache')
    else:
      directory = os.path.join(directory , 'aa_ppi_cache')
    if not os.path.exists(directory):
      os.mkdir(directory)

    cached_features_file = os.path.join(directory, f'cached_lm_{block_size}_{filename}')

    if os.path.exists(cached_features_file+'label1hot'): ## take 1 thing to test if it exists
      logger.info("Loading features from cached file %s", cached_features_file)
      with open(cached_features_file+'label1hot', 'rb') as handle:
        self.label1hot = pickle.load(handle)
      with open(cached_features_file+'input_ids_aa', 'rb') as handle:
        self.input_ids_aa = pickle.load(handle)
      with open(cached_features_file+'input_ids_label', 'rb') as handle:
        self.input_ids_label = pickle.load(handle)
      with open(cached_features_file+'mask_ids_aa', 'rb') as handle:
        self.mask_ids_aa = pickle.load(handle)
      with open(cached_features_file+'ppi_vec', 'rb') as handle:
        self.ppi_vec = pickle.load(handle)
      if args.aa_type_emb:
        with open(cached_features_file+'aa_type_emb', 'rb') as handle:
          self.aa_type_emb = pickle.load(handle)

    else:

      annot_name_sorted = None
      if args.aa_type_file is not None:
        print ('load in aa_type_file')
        annot_data = pickle.load ( open ( args.aa_type_file, 'rb' ) )
        temp = sorted (list (annot_data.keys() ) ) ## easiest to just do by alphabet, so we can backtrack very easily
        ## make sure we exclude position which is last. @annot_name_sorted is index-lookup, so we have +2
        annot_name_sorted = {value: index+2 for index, value in enumerate(temp)} ## lookup index


      label_2test_array = sorted(label_2test_array) ## just to be sure we keep alphabet
      num_label = len(label_2test_array)
      print ('num label {}'.format(num_label))
      label_index_map = { name : index for index,name in enumerate(label_2test_array) } ## faster look up
      # label_string = " ".join(label_2test_array)

      logger.info("Creating features from dataset file at %s", directory)

      self.label1hot = [] ## take 1 hot (should short labels by alphabet)
      self.input_ids_aa = []
      self.input_ids_label = []
      self.mask_ids_aa = []
      self.ppi_vec = [] ## some vector on the prot-prot interaction network... or something like that
      if args.aa_type_emb:
        self.aa_type_emb = []

      fin = open(file_path,"r",encoding='utf-8')
      for counter, text in tqdm(enumerate(fin)):

        # if counter > 100 :
        #   break

        text = text.strip()
        if len(text) == 0: ## skip blank ??
          continue

        ## we test all the labels in 1 single call, so we have to always get all the labels.
        ## notice we shift 1+ so that we can have padding at 0.
        # self.input_ids_label.append ( list((1+np.arange(num_label+1))) )  ## add a SEP to end of label side ??? Okay, add SEP
        self.input_ids_label.append ( np.arange(num_label).tolist() )  ## add a SEP to end of label side ??? Okay, add SEP

        ## split at \t ?? [seq \t label]
        text = text.split("\t") ## position 0 is kmer sequence, position 1 is list of labels

        ### !!!!
        ### !!!! now we append the protein-network vector
        self.ppi_vec.append ([float(s) for s in text[3].split()]) ## 3rd tab

        ## create a gold-standard label 1-hot vector.
        ## convert label into 1-hot style
        label1hot = np.zeros(num_label) ## 1D array
        this_label = text[2].strip().split() ## by space
        index_as1 = [label_index_map[label] for label in this_label]
        label1hot [ index_as1 ] = 1
        self.label1hot.append( label1hot )

        # kmer_text = text[0].split() ## !! we must not use string text, otherwise, we will get wrong len
        ## GET THE AA INDEXING '[CLS] ' + text[0] + ' [SEP]'
        this_aa = tokenizer.convert_tokens_to_ids ( tokenizer.tokenize ('[CLS] ' + text[1] + ' [SEP]') )
        len_withClsSep = len(this_aa)

        ## pad @this_aa to max len
        mask_aa = [1] * len_withClsSep + [0] * ( max_aa_len - len_withClsSep ) ## attend to non-pad
        this_aa = this_aa + [0] * ( max_aa_len - len_withClsSep ) ## padding
        self.input_ids_aa.append( this_aa )
        self.mask_ids_aa.append (mask_aa)

        if args.aa_type_emb:
          ### !!! need to get token type emb of AA in protein
          ## in evaluation mode, do not need to random assign UNK
          AA = ReadProtData(text[4],len_withClsSep-2,max_aa_len,annot_data,annot_name_sorted,evaluate=evaluate)
          self.aa_type_emb.append ( AA )

        if counter < 3:
          print ('see sample {}'.format(counter))
          print (this_aa)
          print (label1hot)
          print (self.ppi_vec[counter])

        if (len(this_aa) + num_label) > block_size:
          print ('len too long, expand block_size')
          exit()

      ## save at end
      logger.info("To save read/write time... Saving features into cached file %s", cached_features_file)
      with open(cached_features_file+'label1hot', 'wb') as handle:
        pickle.dump(self.label1hot, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'input_ids_aa', 'wb') as handle:
        pickle.dump(self.input_ids_aa, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'input_ids_label', 'wb') as handle:
        pickle.dump(self.input_ids_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'mask_ids_aa', 'wb') as handle:
        pickle.dump(self.mask_ids_aa, handle, protocol=pickle.HIGHEST_PROTOCOL)
      with open(cached_features_file+'ppi_vec', 'wb') as handle:
          pickle.dump(self.ppi_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

      if args.aa_type_emb:
        with open(cached_features_file+'aa_type_emb', 'wb') as handle:
          pickle.dump(self.aa_type_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def __len__(self):
    return len(self.input_ids_aa)

  def __getitem__(self, item):
    if self.args.aa_type_emb:
      return (torch.LongTensor(self.label1hot[item]),
              torch.tensor(self.input_ids_aa[item]),
              torch.tensor(self.input_ids_label[item]),
              torch.tensor(self.mask_ids_aa[item]),
              torch.tensor(self.ppi_vec[item]),
              torch.LongTensor(self.aa_type_emb[item].toarray()))
    else:
      return (torch.LongTensor(self.label1hot[item]),
              torch.tensor(self.input_ids_aa[item]),
              torch.tensor(self.input_ids_label[item]),
              torch.tensor(self.mask_ids_aa[item]),
              torch.tensor(self.ppi_vec[item]) )


def load_and_cache_examples(args, tokenizer, label_2test_array, evaluate=False):
  dataset = TextDataset(tokenizer, label_2test_array, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size, args=args, evaluate=evaluate)
  return dataset


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("--aa_type_file", type=str, default=None)
  parser.add_argument("--pretrained_label_path", type=str, default=None)
  parser.add_argument("--label_2test", type=str, default=None)
  parser.add_argument("--bert_vocab", type=str, default=None)
  parser.add_argument("--config_override", action="store_true")
  parser.add_argument("--aa_type_emb", action="store_true", default=False)

  ## Required parameters
  parser.add_argument("--train_data_file", default=None, type=str, required=True,
            help="The input training data file (a text file).")
  parser.add_argument("--output_dir", default=None, type=str, required=True,
            help="The output directory where the model predictions and checkpoints will be written.")

  parser.add_argument("--eval_data_file", default=None, type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
  parser.add_argument("--config_name", default="", type=str,
            help="Optional pretrained config name or path if not the same as model_name_or_path")
  parser.add_argument("--tokenizer_name", default="", type=str,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
  parser.add_argument("--cache_dir", default="", type=str,
            help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
  parser.add_argument("--block_size", default=-1, type=int,
            help="Optional input sequence length after tokenization."
               "The training dataset will be truncated in block of this size for training."
               "Default to the model max input length for single sentence inputs (take into account special tokens).")
  parser.add_argument("--do_train", action='store_true',
            help="Whether to run training.")
  parser.add_argument("--do_eval", action='store_true',
            help="Whether to run eval on the dev set.")
  parser.add_argument("--evaluate_during_training", action='store_true',
            help="Run evaluation during training at each logging step.")
  parser.add_argument("--do_lower_case", action='store_true',
            help="Set this flag if you are using an uncased model.")

  args = parser.parse_args()
  print (args)

  tokenizer = BertTokenizer.from_pretrained(args.bert_vocab, do_lower_case=args.do_lower_case)

  """ model """

  label_2test_array = pd.read_csv(args.label_2test,header=None) # read in labels to be testing
  label_2test_array = sorted(list( label_2test_array[0] ))
  label_2test_array = [re.sub(":","",lab) for lab in label_2test_array] ## splitting has problem with the ":"
  num_labels = len(label_2test_array)

  train_dataset = load_and_cache_examples(args, tokenizer, label_2test_array, evaluate=False)
  eval_dataset = load_and_cache_examples(args, tokenizer, label_2test_array, evaluate=True)


if __name__ == "__main__":
  main()

