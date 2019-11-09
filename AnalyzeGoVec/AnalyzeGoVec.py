
import pickle,sys,os,re
import pandas as pd
import numpy as np
import torch

# we need to see why pretrained does not do well. 
# get go vectors (1st input layer)
# do TSNE or PCA
os.chdir('/local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeEp10e10Drop0.1/checkpoint-90594')

onto_type = 'mf'
model_params = torch.load('pytorch_model.bin') 
## get the GO vec trained jointly with AA sequence
GOvec = model_params['bert.embeddings_label.word_embeddings.weight'].cpu().data.numpy() 
model_params = None ## clear from GPU 
GOvec.shape ## num_label x 256 dim for each go label 

## GO vec are already assigned based on alphabet 
GOname = pd.read_csv('/local/datdb/deepgo/data/train/deepgo.'+onto_type+'.csv',header=None)
GOname = list (GOname[0])

## read IC 
temp = pd.read_csv('/local/datdb/goAndGeneAnnotationMar2017/ICdata/HumanIc'+onto_type.upper()+'.txt', header=None,sep="\t") 
ic_val = dict( zip(list(temp[0]), list(temp[1])) ) 

fout= open ('GOvecFromModel.tsv','w')
for index,value in enumerate(GOname):
  if value in ic_val: 
    ic_is = str(ic_val[value])
  else: 
    ic_is = '0'
  fout.write( value+'\t'+"\t".join(str(v) for v in GOvec[index] )+'\t'+ic_is+'\n' )

#

fout.close() 
