



import sys, re, os, pickle
import pandas as pd
import numpy as np

#### write labels into text, so we can backtrack

#### because we remove some proteins due to super long length, we may not get the same set of GO as in deepgoplus


os.chdir('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000')

ontology = ['cc','mf','bp']
for onto in ontology:
  all_label = {}
  fin = open ('deepgoplus.cafa3.train-bonnie-'+onto+'.tsv','r')
  for index, line in enumerate( fin ) :
    line = line.split('\t')
    labels = line[2].strip().split()
    if 'none' in labels : ## empty label, not all proteins have all 3 categories
      continue
    #
    labels = [re.sub('GO','GO:',l) for l in labels]
    for l in labels:
      if l not in all_label:
        all_label[l] = 1
      else:
        all_label[l] = 1 + all_label[l]
  #
  fin.close()
  #### write out the dict
  fout = open('Label.'+onto+'.tsv','w')
  keys = sorted( list ( all_label.keys() ) ) 
  for key in keys :
    fout.write(key + "\t" + str(all_label[key]) + "\n")
  #
  fout.close()
  #! convert to pickle format. (only needed for deepgoplus style)
  data = {'terms':keys}
  df = pd.DataFrame.from_dict (data) 
  df.to_pickle ('Label.'+onto+'.pickle')

