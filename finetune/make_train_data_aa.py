
import re,sys,os,pickle
import pandas as pd
import numpy as np

np.random.seed(seed=201909) ## use year month as seed

MAX_LEN = 512

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
def split_seq (seq_kmer, split_group=1): ## @seq_kmer is array to make it easier to work with
  where = len(seq_kmer)//split_group
  seq1 = ""
  start = 0
  do_break = False
  for g in range(split_group):
    end = start+where
    if end > len(seq_kmer): ## end point for this group
      end = len(seq_kmer)
      do_break = True
    if len(seq1) == 0:
      seq1 = " ".join( seq_kmer[start:end] )
    else:
      seq1 = seq1 +"\n"+ " ".join( seq_kmer[start:end] )
    ## update @start
    start = start + where ## next place
    if do_break:
      break
  return seq1 ## one sent per line, and blank between "document"


def seq2sentence (seq,kmer_len=3):
  seq_kmer = []
  for i in np.arange(0,len(seq),kmer_len):
    seq_kmer.append ( seq[i:(i+3)] )
  ###
  if len(seq_kmer) >= MAX_LEN:
    seq_kmer = seq_kmer[1:MAX_LEN] ## must shorten
  return split_seq(seq_kmer)


# AA_TYPE_MAP = {}
def aa_type_emb (one_aa):
  # https://proteinstructures.com/Structure/Structure/amino-acids.html
  if one_aa in 'RKDE':
    return 1
  if one_aa in 'QNHSTYC':
    return 2
  if one_aa in 'WYM':
    return 3
  if one_aa in 'AILMFVPG':
    return 4
  return 0


AA_type = {}

for data_type in ['train','dev','test']:
  for ontology in ['cc','bp','mf']:

    long_count = 0

    fin = "/u/scratch/d/datduong/deepgo/data/train/fold_1/"+data_type+"-"+ontology+".tsv"
    fout = "/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify/"+data_type+"-"+ontology+"-aa.csv"

    df = pd.read_csv(fin,sep="\t")
    print (fin)
    print ( df.shape )

    ## write out each line, no header and make into kmer
    # Entry Gene ontology IDs Sequence  Prot Emb

    fout = open(fout,'w')
    for index,row in df.iterrows():
      if (len(row['Sequence'])> 1024):
        long_count = long_count+1

      new_seq = row['Sequence']

      for aa in new_seq:
        if aa not in AA_type:
          AA_type[aa] = 1
        else:
          pass

      aa_type = [aa_type_emb(aa) for aa in new_seq]
      go_list = re.sub(r":","",row['Gene ontology IDs'])
      go_list = sorted(go_list.split(";"))
      fout.write(" ".join(new_seq) + "\t" + " ".join(go_list)+ "\t" + " ".join(row['Prot Emb'].strip().split(';')) + "\t"+' '.join(str(aa) for aa in aa_type) + "\n")

    fout.close()
    print ('long seq counter {}'.format(long_count))


##
print (len(AA_type))

