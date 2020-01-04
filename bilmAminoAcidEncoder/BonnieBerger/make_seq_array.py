

import pickle, os, sys, re
import pandas as pd
import numpy as np

os.chdir("/local/datdb/deepgo/data/train/fold_1/ProtAnnotTypeData")
## create an array in the exact order as file
for data_type in ['test','train','dev']:
  for onto in ['cc','bp','mf']:
    prot_order = []
    seq_order = []
    fin = open(data_type+"-"+onto+"-input.tsv","r")
    for line in fin:
      line = line.strip().split("\t")
      prot_order.append (line[0])
      seq_order.append (line[1].split())
    #
    fin.close() 
    pickle.dump( [prot_order,seq_order], open(data_type+"-"+onto+"-sequence-array.pickle","wb"))


# z = pickle.load( open(data_type+"-"+onto+"-sequence-array.pickle","rb"))
