

import pickle,sys,os,re
import pandas as pd
import numpy as np
import torch

import networkx
import obonet

# pretrained_label=pickle.load(open('/u/scratch/d/datduong/deepgo/data/cosine.AveWordClsSep768.Linear256.Layer12/label_vector.pickle','rb'))


work_dir = '/u/scratch/d/datduong/deepgo/data/'
os.chdir (work_dir)
# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
# Mapping from term ID to name
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')} ## by default obsolete already removed
# id_to_name['GO:0000002']

# we need to see why pretrained does not do well.
# get go vectors (1st input layer)
# do TSNE or PCA

this_list = {'mf': ['GO:0030291', 'GO:0000980', 'GO:1905030', 'GO:0004653'],
             'bp': ['GO:0051052', 'GO:0018193', 'GO:0097352', 'GO:0045666'],
             'cc': ['GO:0005743', 'GO:0098791']}

# onto_type_dict = {'mf': 90594, 'bp': 116144, 'cc': 127647}
onto_type_dict = {'mf': 80528, 'bp': 145180, 'cc': 141830}

for onto_type,onto_checkpoint in onto_type_dict.items() :

  # if onto_type != 'bp':
  #   continue

  # P08090  GO:0030291
  # GO:0000980

  color = {}
  for t in this_list[onto_type]:
    color[t] = sorted ( [t] + list ( networkx.descendants(graph, t ) ) ) ## return parents https://www.ebi.ac.uk/QuickGO/term/GO:0000002

  os.chdir('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'+onto_type+'/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/')

  GOname = pd.read_csv('/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/deepgo.'+onto_type+'.csv',header=None)
  GOname = GOname.sort_values(by=[0],inplace=False).reset_index(drop=True)
  GOname_array = list (GOname[0])
  ## remove terms not used in deepgo
  GOname[1] = 0
  for index, t in enumerate(this_list[onto_type]):
    GOname.loc[GOname[0].isin(color[t]),1] = index+1

  color = list ( GOname[1] )

  ## read IC
  temp = pd.read_csv('/u/scratch/d/datduong/goAndGeneAnnotationMar2017/ICdata/HumanIc'+onto_type.upper()+'.txt', header=None,sep="\t")
  ic_val = dict( zip(list(temp[0]), list(temp[1])) )

  ## write ic to GO vec
  fin = open ('GOvecFromModelHiddenLayer12Expandtest.tsv','r') ## should already in alphabet order
  fout= open ('GOvecFromModelHiddenLayer12ExpandtestWithIc.tsv','w')
  
  for index,line in enumerate(fin):
    line = line.strip()
    value = line.split()[0]
    if value in ic_val:
      ic_is = str(ic_val[value])
    else:
      ic_is = '0'
    fout.write( line+'\t'+ic_is+'\t'+str(color[index])+'\n' )

  #
  fin.close()
  fout.close()

