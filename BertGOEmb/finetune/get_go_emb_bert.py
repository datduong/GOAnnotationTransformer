
from argparse import ArgumentParser
from pathlib import Path
import os, pickle, sys
import torch
import logging
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME, BertConfig
from pytorch_transformers.modeling_bert import BertForPreTraining
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule



## we used the GO vectors as "word"
## we can get the GO emb directly from the BERT model 

output = "" 
bert_model = "" 
word_path = ""

model = BertForPreTraining.from_pretrained(bert_model)
word_emb = model.bert.embeddings.data.numpy() ## extract as np

word_text = pd.read_csv(word_path,header=None,sep="\t") ## read in the actual word, notice, the ordering here matches the exact ordering of @word_emb
word_text = list (word_text[0])

word_dict = {}
for index, word in enumerate(word_text): 
  word_dict[word_text] = word_emb[index]


pickle.dump( word_dict, open(output,"wb") ) 


