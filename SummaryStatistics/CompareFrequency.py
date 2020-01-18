

import os,sys,re,pickle
import numpy as np
import pandas as pd 

#### count number of label for each protein

#### frequency per GO labels


for onto in ['mf','cc','bp']:
  LabelCount = {}
  NumLabelPerSample = []
  # dataExpandGoSet
  # file_name = "/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/train-"+onto+"-same-origin.tsv"
  #### call our train partition
  file_name = "/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/ProtAnnotTypeData/train-"+onto+"-input.tsv"
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
  ##
  #### original file
  LabelCountOriginal = {}
  file_name = "/local/datdb/deepgo/data/train/fold_1/train-"+onto+".tsv" 
  fin = open(file_name,"r")
  for index,line in enumerate(fin):
    if index == 0:
      continue
    line = line.strip().split('\t')
    line[1] = re.sub(":","",line[1])
    label = line[1].split(";")
    # NumLabelPerSample.append ( len(label) )
    for l in label:
      if l in LabelCountOriginal:
        LabelCountOriginal[l] = LabelCountOriginal[l] + 1
      else:
        LabelCountOriginal[l] = 1
  fin.close()
  ## compare: 
  print ('\n\nonto {}'.format(onto))
  if onto == 'mf': 
    print ('see example')
    print (LabelCountOriginal['GO0004930'])
    print (LabelCount['GO0004930'])
  print_see_more_counter = 0 
  for k,v in LabelCountOriginal.items(): 
    if LabelCount[k] > v: 
      print_see_more_counter = print_see_more_counter + 1
  #
  print ('print_see_more_counter')
  print (print_see_more_counter)

