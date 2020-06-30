

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


#### ? ordering of the @term matters

terms = list( pd.read_pickle('terms.pkl')['terms'] ) #! ordering matters for @terms
label_in_onto = []
col_keep = []
for index, label in enumerate(terms):
  if graph.node[label]['namespace'] == onto:
    col_keep.append(index)
    label_in_onto.append(label)


print ('total num label in ontology {}'.format(len(label_in_onto)))

df = pd.read_pickle('predictions.pkl')

label_np = np.zeros( (df.shape[0],677) ) #! 677 mf labels
prediction_np = np.zeros( (df.shape[0],677) ) #! 677 mf labels

label_to_keep = [] ##! these are for new dataframe
label_np_to_keep = []
prediction_to_keep = []

for index,row in df.iterrows():
  label_np [index] = row['labels'][col_keep]
  label_np_to_keep.append(row['labels'][col_keep])
  prediction_np [index] = row['preds'][col_keep]
  prediction_to_keep.append(row['preds'][col_keep])
  # this_label = [ label for label in row['annotations'] if label in label_in_onto ] # ! ! very important to not filter the labels here. not use [col_keep]
  # label_to_keep.append( set( this_label ) )
  label_to_keep.append( row['annotations'] )

#

output = {'prediction':prediction_np, 'true_label':label_np}

pickle.dump(output, open('predictions.numpy.pickle','wb'))

df['annotations'] = label_to_keep
df['labels'] = label_np_to_keep
df['preds'] = prediction_to_keep

df.to_pickle('predictions_filter_by_mf.pkl')

df



# {GO:0043296, GO:0030054, GO:0030031, GO:000382...
# labels         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
