
import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm




def seq2sentence (seq,kmer_len=3):
  seq_kmer = []
  for i in np.arange(0,len(seq),kmer_len):
    seq_kmer.append ( seq[i:(i+3)] ) ## split by 3mer
  ###
  if len(seq_kmer) >= 512:
    seq_kmer = seq_kmer[1:512] ## must shorten
  return " ".join(seq_kmer)
  


## just do sent1 sent2 style 

fout = open("","w")
fout.write("index\tquestion\tsentence\tlabel\n")

# Entry Entry name  Status  Gene names  Organism  Length  Gene ontology IDs Sequence  Protein names Organism ID

uniprot = pd.read_csv("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/uniprot-filtered-reviewed_yes.tab",sep="\t",dtype=str)

for index,row in tqdm(uniprot.iterrows()): 
  largest_len_divisible = int ( np.floor ( len(row['Sequence']) / 3 ) )
  new_seq = row['Sequence'][0:largest_len_divisible]
  new_seq = seq2sentence (new_seq)

