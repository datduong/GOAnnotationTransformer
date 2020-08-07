

##! we use original data to get the 3d vec.
##! need new code to add the 3d vec new data

import sys,os,re,pickle
import numpy as np
import pandas as pd

data_dir = '/u/scratch/d/datduong/deepgoplus/ExpandGoSet/cafa3-data/'

os.chdir('/u/scratch/d/datduong/deepgoplus/ExpandGoSet/cafa3-data/SeqLenLess1000')

for data_type in ['train','test'] :

  for onto in ['mf','cc']:

    #! read in original data
    FullLenWith3dVec = pd.read_csv ('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/FullLen/deepgoplus.cafa3.'+data_type+'-bonnie-'+onto+'.tsv',sep='\t',header=None)
    Map3dVec = {}
    for index,row in FullLenWith3dVec.iterrows():
      Map3dVec [ row[0] ] = row[3]

    ##! read in new expanded data. add in vector. format them
    fin = pickle.load (open(data_dir+data_type+'-'+onto+'.pkl','rb'))
    fout = open('deepgoplus.cafa3.'+data_type+'-bonnie-'+onto+'.tsv','w')

    for index,line in fin.iterrows():
      if len(line['sequences']) > 1000:
        continue
      seq = " ".join(i for i in line['sequences']) # spacing
      terms = [ re.sub ( ':', '', g) for g in line['annotations'] ] # remove ":" make life easier later
      terms = ' '.join( terms )
      if len (terms) == 0:
        terms = 'none'
      newline = line['proteins'] + '\t' + seq + '\t' + terms + '\t' + Map3dVec[line['proteins']] + '\n'
      fout.write(newline)

    #
    fout.close()
