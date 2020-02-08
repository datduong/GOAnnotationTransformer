
import sys,re,os,pickle
import numpy as np
import pandas as pd

#### split using int(n * 0.8)
def train_validate_test_split(df, train_percent=.80, seed=1234):
  np.random.seed(seed)
  perm = np.random.permutation(df.index)
  m = len(df.index)
  train_end = int(train_percent * m)
  train = df.ix[perm[:train_end]]
  validate = df.ix[perm[train_end:]]
  return train, validate


np.random.seed(0) ####

main_dir = '/u/scratch/d/datduong/UniprotJan2020/AddIsA/'
os.chdir(main_dir)

TrainDevTest = 'TrainDevTest/'

for onto in ['mf','cc','bp']:
  test_set = pickle.load(open(onto+"_in_test_name.pickle","rb")) # cc_in_train_name.pickle
  train_set = pickle.load(open(onto+"_in_train_name.pickle","rb"))
  test_fout = open(TrainDevTest+onto+"-test.tsv","w")
  train_fout = open(TrainDevTest+onto+"-train.tsv","w")
  dev_fout = open(TrainDevTest+onto+"-dev.tsv","w")
  fin = open(onto+"-input.tsv","r") # cc-input.tsv
  for index,line in enumerate(fin):
    name = line.split('\t')[0]
    if name in test_set:
      test_fout.write(line) ## no strip, so we leave it as it is
      continue
    if name in train_set:
      train_fout.write(line)
      continue
    sample_random = np.random.uniform(low=0.0, high=1.0)
    if sample_random <=.2: ## 20% test
      test_fout.write(line)
    if (sample_random <= .3) and (sample_random>.2): ## 10% dev
      dev_fout.write(line)
    if sample_random > .3:
      train_fout.write(line)
  #
  fin.close()
  test_fout.close()
  train_fout.close()
  dev_fout.close()



