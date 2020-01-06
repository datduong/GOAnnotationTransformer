import os,sys,re,pickle
import pandas as pd
import numpy as np

#### see if we have correct number of labels

num_label = {} 

df = pd.read_csv("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData/train-bp-input-muhao.tsv",header=None,sep='\t')
for index,row in df.iterrows():
  label = row[2].strip().split()
  for l in label:
    if l not in num_label:
      num_label[l] = 1


print ('len {}'.format(len(num_label)))
