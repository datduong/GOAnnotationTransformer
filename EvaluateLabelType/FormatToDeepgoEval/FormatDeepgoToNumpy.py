

import pickle,os,sys,re
import pandas as pd
import numpy as np

import networkx
import obonet


####

os.chdir('/u/scratch/d/datduong/deepgoplus/data-cafa')

onto = 'molecular_function'

##!! need debug to see why fmax changes.

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
# graph.node['GO:0003824']['namespace']

terms = list( pd.read_pickle('terms.pkl')['terms'] )
col_keep = []
for index, label in enumerate(terms):
  if graph.node[label]['namespace'] == onto:
    col_keep.append(index)



df = pd.read_pickle('predictions.pkl')

label_np = np.zeros( (df.shape[0],677) ) #! 677 mf labels
prediction_np = np.zeros( (df.shape[0],677) ) #! 677 mf labels

for index,row in df.iterrows():
  label_np [index] = row['labels'][col_keep]
  prediction_np [index] = row['preds'][col_keep]


#

output = {'prediction':prediction_np, 'true_label':label_np}

pickle.dump(output, open('predictions.numpy.pickle','wb'))


# {GO:0043296, GO:0030054, GO:0030031, GO:000382...
# labels         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
