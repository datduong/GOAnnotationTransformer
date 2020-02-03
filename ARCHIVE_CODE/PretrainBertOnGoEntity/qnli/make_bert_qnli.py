
import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy



def seq2sentence (seq,kmer_len=3):
  seq_kmer = []
  for i in np.arange(0,len(seq),kmer_len):
    seq_kmer.append ( seq[i:(i+3)] ) ## split by 3mer
  ###
  if len(seq_kmer) >= 512:
    seq_kmer = seq_kmer[1:512] ## must shorten
  return " ".join(seq_kmer)
  

## get list of GO terms so we can create "not entailment"
GO_list = pd.read_csv("",sep="")
GO_list = list(GO_list[''])

## sample random go not in the list 
def rand_sample_go (GO_list,go_remove): 
  GO2 = deepcopy(GO_list)
  for g in go_remove: 
    GO2.remove(g)
  rand_size = len(go_remove)
  if rand_size < 10: 
    rand_size = 10 
  GO = np.random.choice(GO2,size=rand_size,replace=False)
  return " ".join(GO)

## just do sent1 sent2 style 

full_counter = 0 
fout = open("","w")
fout.write("index\tquestion\tsentence\tlabel\n")

# Entry Entry name  Status  Gene names  Organism  Length  Gene ontology IDs Sequence  Protein names Organism ID

uniprot = pd.read_csv("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/uniprot-filtered-reviewed_yes.tab",sep="\t",dtype=str)

for index,row in tqdm(uniprot.iterrows()): 
  seq = row['Sequence'][ 1:(len(row['Sequence'])-1) ] ## remove start/stop codon ?
  largest_len_divisible = int ( np.floor ( len(seq) / 3 ) ) * 3
  new_seq = seq[0:largest_len_divisible]
  new_seq = seq2sentence (new_seq)
  GO_assign = row['Gene ontology IDs'].split(';')
  fout.write(str(full_counter)+"\t"+new_seq+"\t"+" ".join( GO_assign ) + "\tentailment\n" )
  full_counter = full_counter + 1 
  GO_rand = rand_sample_go(GO_list, GO_assign) ## get random set of GO not related to this sequence 
  fout.write(str(full_counter)+"\t"+new_seq+"\t"+" ".join( GO_rand ) + "\tnot_entailment\n" )
  full_counter = full_counter + 1 

## read GO pairs 

GO_pairs = pd.read_csv("",sep="\t")
for index,row in GO_pairs.iterrows(): 
  if GO_pairs['label'] == 'not_entailment': 
    fout.write(str(full_counter)+"\t"+GO_pairs['go1']+"\t"+GO_pairs['go2'] + "\tnot_entailment\n" )
    full_counter = full_counter + 1 
    fout.write(str(full_counter)+"\t"+GO_pairs['go2']+"\t"+GO_pairs['go1'] + "\tnot_entailment\n" )
    full_counter = full_counter + 1 

## add axiom pairs ?? 

# fout = open ("/u/scratch/d/datduong/UniprotAllReviewGoAnnot/go_qnli.txt","w")
go_data = pd.read_csv("/u/scratch/d/datduong/Onto2Vec/GOVectorData/2017/AllAxioms.lst",dtype=str,sep="|",header=None) ## doesnt' matter what sep
## add go terms into fine tune as well 
for index,row in tqdm (go_data.iterrows()): 
  line = row[0].split() 
  has_go = np.array ( [bool(re.match('GO_',j)) for j in line] ) 
  if np.sum ( has_go ) > 1: 
    # where_replace = np.where(has_go==True)[0]
    ## remove _ with : to get GO:xyz
    fout.write (str(full_counter)+"\t"+re.sub("_",":",line[0]) + "\t" + re.sub ("_",":", " ".join(line[1::])) + "\tentailment\n")
    full_counter = full_counter + 1 


fout.close() 




  

