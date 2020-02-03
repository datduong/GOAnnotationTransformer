

## follow paper by bonnie berger lab https://arxiv.org/pdf/1902.08661.pdf

from __future__ import print_function,division

import pickle, os, sys, re
import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import torch.utils.data

sys.path.append('/local/datdb/ProteinEmbMethodGithub/protein-sequence-embedding-iclr2019/')

from src.alphabets import Uniprot21
import src.scop as scop
from src.utils import pack_sequences, unpack_sequences
from src.utils import PairedDataset, AllPairsDataset, collate_paired_sequences
from src.utils import MultinomialResample
import src.models.embedding
import src.models.comparison


model = torch.load("/local/datdb/ProteinEmbMethodGithub/protein-sequence-embedding-iclr2019/pretrained_models/me_L1_100d_lstm3x512_lm_i512_mb64_tau0.5_p0.05_epoch100.sav")

model.eval()
model.cuda()

alphabet = Uniprot21() ##!! convert string to indexing.

def submitJobs (onto) :

  os.chdir("/local/datdb/UniprotJan2020")
  ## create an array in the exact order as file
  for onto in [onto] : # ['cc','bp','mf']:
    fin = open("uniprot-"+onto+".tsv","r") # Entry   Gene ontology IDs       Sequence
    fout = open("uniprot-"+onto+"-bonnie.tsv","w")
    for index, line in tqdm (enumerate(fin)): #### retain the same ordering as original input
      if index == 0:
        fout.write(line.strip()+"\tprot3dvec\n") ## header
        continue ##!! skip header
      line = line.strip().split("\t")
      seq = str.encode (re.sub(" ","",line[2])) ## no spacing in string
      in_index = Variable(torch.LongTensor([alphabet.encode(seq)])).cuda()
      vec = model.embedding ( in_index )
      vec = torch.mean(vec,1).cpu().data.numpy()[0] ## just do per seq
      line_len = len(line) ## keep everything same as before.
      new_line = "\t".join( line[j] for j in range(line_len) ) + "\t" + " ".join(str(s) for s in vec) + "\n" # "\t" + line[line_len+1]
      fout.write(new_line)
    #
    fin.close()
    fout.close()


# z = pickle.load( open(data_type+"-"+onto+"-sequence-array.pickle","rb"))

if len(sys.argv)<1: ## run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] )



