

import os,sys,re,pickle
import numpy as np

#### count number of label for each protein

#### frequency per GO labels


for onto in ['mf','cc','bp']:
  LabelCount = {}
  NumLabelPerSample = []
  # dataExpandGoSet
  file_name = "/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData/train-"+onto+"-input-bonnie.tsv"
  fin = open(file_name,"r")
  for index,line in enumerate(fin):
    # if index == 0:
    #   continue
    line = line.strip().split('\t')
    label = line[2].split(" ")
    NumLabelPerSample.append ( len(label) )
    for l in label:
      if l in LabelCount:
        LabelCount[l] = LabelCount[l] + 1
      else:
        LabelCount[l] = 1
  fin.close()
  print ('\nonto {}'.format(onto))
  print ('NumLabelPerSample')
  print ( np.quantile(NumLabelPerSample,q=[0.25,.5,.75,.95,1]) )
  print ('GO label counter')
  counter = [v for k,v in LabelCount.items()]
  print ( np.quantile(counter,q=[0.25,.5,.75,.95,1]) )


