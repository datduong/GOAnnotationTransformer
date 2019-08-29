

import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm 

## make bert finetune data

## use the kmer-as vocab.

## GO vector need a vocab too ?? no. we can just replace the embedding ??

## for each sequence, we split them into chunks of len 30 (10 words ??)
## for the same sequence, we also split half chunk ?? so we can learn position emb ??

# use these amino acids ARNDCQEGHILKMFPSTWYVXOUBZ
## use 3kmer so we get 8000, and then we can replace BERT vocab
kmer_vocab = {}
counter = 0
string = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
for a1 in string:
  for a2 in string:
    for a3 in string:
      kmer_vocab[a1+a2+a3] = counter ## record 1:abc so that we can use indexing count later
      counter = counter + 1

## split
def split_seq (seq_kmer): ## @seq_kmer is array to make it easier to work with
  where = len(seq_kmer)//2
  seq1 = " ".join( seq_kmer[0:where] )
  seq2 = " ".join( seq_kmer[where::] )
  return seq1 + "\n" + seq2 ## one sent per line, and blank between "document"


def seq2sentence (seq,kmer_len=3):
  seq_kmer = []
  for i in np.arange(0,len(seq),kmer_len):
    seq_kmer.append ( seq[i:(i+3)] )

  ###
  if len(seq_kmer) >= 512:
    seq_kmer = seq_kmer[1:512] ## must shorten

  return split_seq(seq_kmer)
  


seq_data = pd.read_csv("/u/scratch/d/datduong/deepgo/data/embeddings.tsv",dtype=str,sep="\t")
# Entry Gene ontology IDs Sequence  Prot Emb
fout = open("/u/scratch/d/datduong/deepgo/data/embeddings_finetune.txt","w")
for index,row in tqdm (seq_data.iterrows()):
  seq = row['Sequence'][ 1:len((row['Sequence'])-1) ] ## remove start/stop codon ?
  largest_len_divisible = int ( np.floor ( len(seq) / 3 ) ) * 3
  new_seq = seq[0:largest_len_divisible]
  new_seq = seq2sentence (new_seq)
  fout.write(new_seq + "\n\n")


fout.close() 


##
fout = open ("/u/scratch/d/datduong/deepgo/data/go_finetune.txt","w")
go_data = pd.read_csv("/u/scratch/d/datduong/Onto2Vec/GOVectorData/2016DeepGOData/AllAxioms_2016.lst",dtype=str,sep="|",header=None) ## doesnt' matter what sep
## add go terms into fine tune as well 
for index,row in tqdm (go_data.iterrows()): 
  line = row[0].split() 
  has_go = np.array ( [bool(re.match('GO_',j)) for j in line] ) 
  if np.sum ( has_go ) > 1: 
    # where_replace = np.where(has_go==True)[0]
    ## remove _ with : to get GO:xyz
    fout.write (re.sub("_",":",line[0]) + "\n" + re.sub ("_",":", " ".join(line[1::])) + "\n\n")


fout.close() 

