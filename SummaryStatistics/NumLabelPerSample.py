

import os,sys,re,pickle
import numpy as np
import pandas as pd 

#### count number of label for each protein

#### frequency per GO labels


for onto in ['mf','cc','bp']:
  LabelCount = {}
  NumLabelPerSample = []
  # dataExpandGoSet
  file_name = "/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/train-"+onto+"-16Jan20.tsv"
  fin = open(file_name,"r")
  for index,line in enumerate(fin):
    # if index == 0:
    #   continue
    line = line.strip().split('\t')
    line[1] = re.sub(":","",line[1])
    label = line[1].split(";")
    NumLabelPerSample.append ( len(label) )
    for l in label:
      if l in LabelCount:
        LabelCount[l] = LabelCount[l] + 1
      else:
        LabelCount[l] = 1
  fin.close()
  print ('\nonto {}'.format(onto))
  print ('Total num label')
  print (len(LabelCount))
  print ('NumLabelPerSample')
  print ( np.quantile(NumLabelPerSample,q=[0.25,.5,.75,.95,1]) )
  print ('GO label counter')
  counter = [v for k,v in LabelCount.items()]
  print ( np.quantile(counter,q=[0.25,.5,.75,.95,1]) )
  #### load in original label
  label_original = pd.read_csv('/u/scratch/d/datduong/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  temp = list(label_original[0])
  label_original = [re.sub(":","",l) for l in temp]
  label_original = set(label_original)
  label_in_dict = set(list(LabelCount.keys()))
  added_label = list (label_in_dict - label_original)
  print ('Added, unseen GO label counter')
  counter = [v for k,v in LabelCount.items() if k in added_label]
  print ( np.quantile(counter,q=[0.05,0.25,.5,.75,.95,1]) )
  print ('Original, seen GO label counter')
  counter = [v for k,v in LabelCount.items() if k in label_original]
  print ( np.quantile(counter,q=[0.05,0.25,.5,.75,.95,1]) )

