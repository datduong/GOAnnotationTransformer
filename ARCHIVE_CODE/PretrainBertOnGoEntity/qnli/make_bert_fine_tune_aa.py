

import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm 


np.random.seed(seed=201909) ## use year month as seed

MAX_LEN = 1022

## make bert finetune data

## use the kmer-as vocab.

## GO vector need a vocab too ?? no. we can just replace the embedding ??

## for each sequence, we split them into chunks of len 30 (10 words ??)
## for the same sequence, we also split half chunk ?? so we can learn position emb ??

# use these amino acids ARNDCQEGHILKMFPSTWYVXOUBZ

counter = 0
string = 'ARNDCQEGHILKMFPSTWYVXOUBZ'


# seq_data = pd.read_csv("/u/scratch/d/datduong/deepgo/data/embeddings.tsv",dtype=str,sep="\t")
seq_data = pd.read_csv("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/uniprot-filtered-reviewed_yes.tab",dtype=str,sep="\t")
# Entry Gene ontology IDs Sequence  Prot Emb
fout = open("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/seq_finetune_aa_train.txt","w")

fout2 = open("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/seq_finetune_aa_test.txt","w")

for index,row in tqdm (seq_data.iterrows()):

  if len(row['Sequence']) < 32 : ## not long enough 
    continue 

  if np.random.uniform() > .25 : 
    continue ## take only some data

  seq = row['Sequence'] # [ 1:(len(row['Sequence'])-1) ] ## remove start/stop codon ?
  if len(seq) > MAX_LEN : ## bound
    where = np.random.choice ( np.arange(len(seq)-MAX_LEN), size=1 )
    seq = seq[int(where):MAX_LEN] ## get the random segment 
  
  if np.random.uniform() > .10 : 
    fout.write(seq + "\n\n")
  else: 
    fout2.write(seq + "\n\n") ## testing data

fout.close() 
fout2.close() 


# ##
# fout = open ("/u/scratch/d/datduong/deepgo/data/go_finetune_axiom_train.txt","w")
# fout2 = open ("/u/scratch/d/datduong/deepgo/data/go_finetune_axiom_test.txt","w")
# # fout = open ("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/go_finetune.txt","w")
# go_data = pd.read_csv("/u/scratch/d/datduong/Onto2Vec/GOVectorData/2016DeepGOData/AllAxioms_2016.lst",dtype=str,sep="|",header=None) ## doesnt' matter what sep
# ## add go terms into fine tune as well 
# for index,row in tqdm (go_data.iterrows()): 
#   line = row[0].split() 
#   has_go = np.array ( [bool(re.match('GO_',j)) for j in line] ) 
#   if np.sum ( has_go ) > 1: 
#     # where_replace = np.where(has_go==True)[0]
#     ## remove _ with : to get GO:xyz
#     if np.random.uniform() > .10 : 
#       fout.write (re.sub("_",":",line[0]) + "\n" + re.sub ("_",":", " ".join(line[1::])) + "\n\n")
#     else: 
#       fout2.write (re.sub("_",":",line[0]) + "\n" + re.sub ("_",":", " ".join(line[1::])) + "\n\n")

# fout.close() 
# fout2.close() 
