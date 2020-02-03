

import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd


## make bert finetune data
## use the kmer-as vocab.


os.chdir("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12Kmer/")

# use these amino acids ARNDCQEGHILKMFPSTWYVXOUBZ
## use 3kmer so we get 8000, and then we can replace BERT vocab 
kmer_vocab = {}
counter = 0
string = 'ARNDCQEGHILKMFPSTWYVXOUBZ'
for a1 in string: 
  for a2 in string: 
    for a3 in string: 
      kmer_vocab[counter] = a1+a2+a3 ## record 1:abc so that we can use indexing count later
      counter = counter + 1 


#
pickle.dump(kmer_vocab,open("kmer_vocab.pickle","wb"))


## replace vocab 
## vocab seen 
vocab_came_with_bert = pd.read_csv("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12/vocab.txt",dtype=str,sep="\t",header=None)
vocab_came_with_bert = list(vocab_came_with_bert[0])

fin = open("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12/vocab.txt","r")
fout = open ("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12Kmer/vocab+3kmer.txt","w")

counter_outter = 0 
for line in fin: 
  if (counter_outter<len(kmer_vocab)): 
    if (line.strip() not in '[UNK] [CLS] [SEP] [MASK] [PAD]'.split()): 
      for counter in range(counter_outter,len(kmer_vocab)): 
        if kmer_vocab[counter] in vocab_came_with_bert: ## if it is in the vocab came with BERT, we don't add it
          pass
        else: 
          fout.write(kmer_vocab[counter]+"\n")
          counter_outter = counter + 1  ## start at next spot 
          break ## break inner for loop
    else: 
      fout.write(line)
  else: 
    fout.write(line)


#
fout.close() 
fin.close() 
