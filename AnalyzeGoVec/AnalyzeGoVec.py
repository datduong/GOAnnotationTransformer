
import pickle,sys,os,re
import pandas as pd
import numpy as np
import torch

import networkx
import obonet

work_dir = '/local/datdb/deepgo/data/'
os.chdir (work_dir)
# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
# Mapping from term ID to name
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')} ## by default obsolete already removed
# id_to_name['GO:0000002']  

# P08090  GO:0030291
# GO:0000980
this_list = ['GO:0030291','GO:0000980']
color = {}
for t in this_list: 
  color[t] = sorted ( [t] + list ( networkx.descendants(graph, t ) ) ) ## return parents https://www.ebi.ac.uk/QuickGO/term/GO:0000002

# we need to see why pretrained does not do well. 
# get go vectors (1st input layer)
# do TSNE or PCA
os.chdir('/local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeEp10e10Drop0.1/checkpoint-90594')

## need to label a branch to plot later. 


onto_type = 'mf'
model_params = torch.load('pytorch_model.bin') 
## get the GO vec trained jointly with AA sequence
GOvec = model_params['bert.embeddings_label.word_embeddings.weight'].cpu().data.numpy() 
model_params = None ## clear from GPU 
GOvec.shape ## num_label x 256 dim for each go label 

## GO vec are already assigned based on alphabet 
GOname = pd.read_csv('/local/datdb/deepgo/data/train/deepgo.'+onto_type+'.csv',header=None)
GOname = GOname.sort_values(by=[0],inplace=False).reset_index(drop=True)

GOname_array = list (GOname[0])

## remove terms not used in deepgo
GOname[1] = 0
for index, t in enumerate(this_list): 
  GOname.loc[GOname[0].isin(color[t]),1] = index+1

color = list ( GOname[1] ) 

## read IC 
temp = pd.read_csv('/local/datdb/goAndGeneAnnotationMar2017/ICdata/HumanIc'+onto_type.upper()+'.txt', header=None,sep="\t") 
ic_val = dict( zip(list(temp[0]), list(temp[1])) ) 

fout= open ('GOvecFromModel.tsv','w')
for index,value in enumerate(GOname_array):
  if value in ic_val: 
    ic_is = str(ic_val[value])
  else: 
    ic_is = '0'
  fout.write( value+'\t'+"\t".join(str(v) for v in GOvec[index] )+'\t'+ic_is+'\t'+str(color[index])+'\n' )

#

fout.close() 


## write the same pretrained GO vec 
pretrained_label=pickle.load(open('/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear256.Layer12/label_vector.pickle','rb'))
fout= open ('GOvecFromBert12.tsv','w')
for index,value in enumerate(GOname_array):
  if value in ic_val: 
    ic_is = str(ic_val[value])
  else: 
    ic_is = '0'
  fout.write( value+'\t'+"\t".join(str(v) for v in pretrained_label[value] )+'\t'+ic_is+'\t'+str(color[index])+'\n' )

#

fout.close() 
