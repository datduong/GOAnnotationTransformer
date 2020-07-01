

import pickle,os,sys,re
import pandas as pd
import numpy as np

import networkx
import obonet


####

os.chdir('/u/scratch/d/datduong/deepgoplus/data-cafa')

onto = 'molecular_function'

##!! test file formatted by us
our_test_df = pd.read_csv('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/test-mf.tsv',header=None,sep='\t')
our_test_protein = list( our_test_df[0] )
num_protein = len(our_test_protein)

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

label_np = np.zeros( (num_protein,677) ) #! 677 mf labels
prediction_np = np.zeros( (num_protein,677) ) #! 677 mf labels

label_to_keep = [] ##! these are for new dataframe
label_np_to_keep = []
prediction_to_keep = []

row_index = 0
for index,row in df.iterrows():
  #
  if row['proteins'] not in our_test_protein: 
    continue #? skip these out-of-ontology
  #
  label_np [row_index] = row['labels'][col_keep]
  label_np_to_keep.append(row['labels'][col_keep])
  prediction_np [row_index] = row['preds'][col_keep]
  prediction_to_keep.append(row['preds'][col_keep])
  # this_label = [ label for label in row['annotations'] if label in label_in_onto ] # ! ! very important to not filter the labels here. not use [col_keep]
  # label_to_keep.append( set( this_label ) )
  label_to_keep.append( row['annotations'] )
  row_index = row_index + 1

#
# remove row that has all zero

output = {'prediction':prediction_np, 'true_label':label_np}

pickle.dump(output, open('/u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.pickle','wb'))

# df['annotations'] = label_to_keep
# df['labels'] = label_np_to_keep
# df['preds'] = prediction_to_keep

# df.to_pickle('predictions_filter_by_mf.pkl')

# df



# {GO:0043296, GO:0030054, GO:0030031, GO:000382...
# labels         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
