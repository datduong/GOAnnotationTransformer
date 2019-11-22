
import os,sys,re,pickle
import pandas as pd 
import numpy as np 

os.chdir('/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1')
for onto in ['cc','mf','bp']:
  df1 = pd.read_csv('test-'+onto+'-same-origin.tsv',sep='\t')
  list1 = set(df1['Entry'])
  df2 = pd.read_csv('ProtAnnotTypeData/test-'+onto+'-prot-annot-input.tsv',sep='\t',header=None)
  list2 = set(df2[0])
  missing = list(list1-list2)
  list1 = list(list1)
  list2 = list(list2)
  print('\ntype {}'.format(onto))
  print (missing)
  print ([list1.index(c) for c in missing]) # ['P42276', 'Q9UI56', 'Q8NDZ9', 'Q13748', 'P04436', 'O42886']

