

import pickle,os,sys,re
import pandas as pd
import numpy as np
from copy import deepcopy
import networkx
import obonet


####

os.chdir('/u/scratch/d/datduong/deepgoplus/data-cafa')

need_remove_root = True
onto = 'molecular_function'
onto_name = 'mf'
output_path = '/u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.pickle'


##!! test file formatted by us
our_test_df = pd.read_csv('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/bonnie+motif/test-mf.tsv',header=None,sep='\t')
our_test_protein = list( our_test_df[0] )
num_protein = len(our_test_protein)


# read in this same go.obo as they did
graph = obonet.read_obo('/u/scratch/d/datduong/deepgoplus/data-cafa/go.obo') # https://github.com/dhimmel/obonet # graph.node['GO:0003824']['namespace']


#### ? ordering of the @term matters

roots = ['GO:0008150','GO:0003674','GO:0005575']

terms = list( pd.read_pickle('/u/scratch/d/datduong/deepgoplus/data-cafa/terms.pkl')['terms'] ) #! ordering matters for @terms, this is their original ordering
label_in_onto = []
col_keep = []
for index, label in enumerate(terms):
  if need_remove_root :
    if label in roots:
      print ('skip {}'.format(index))
      continue #? skip the root
  if graph.node[label]['namespace'] == onto:
    col_keep.append(index)
    label_in_onto.append(label)


####

#! need to reorder
our_test_label = pd.read_csv('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'+onto_name+'.tsv',header=None,sep='\t') ##!! do not need to reorder, we already sorted by names
our_test_label = list (our_test_label[0]) ## panda format, take first col
if need_remove_root:
  our_test_label = [lab for lab in our_test_label if lab not in roots] ## remove roots

##!! reorder
fix_their_term_to_our = { our_test_label.index(t):index for index,t in enumerate(label_in_onto) } ## our-->their

def reorder_array (array,fix_their_term_to_our):
  output = np.array(deepcopy(array))
  for i in range(len(array)):
    output[ i ] = array [ fix_their_term_to_our[i] ] ## i=0, find our [0] wrt their @array, then take that value
  return output

# z = reorder_array(label_in_onto,fix_their_term_to_our)

print ('total num label in ontology {}'.format(len(label_in_onto)))

####

num_label = len(our_test_label)
df = pd.read_pickle('predictions.pkl')

label_np = np.zeros( (num_protein,num_label) ) #! 677 mf labels - root
prediction_np = np.zeros( (num_protein,num_label) ) #! 677 mf labels - root

label_to_keep = [] ##! these are for new dataframe
label_np_to_keep = []
prediction_to_keep = []

row_index = 0
for index,row in df.iterrows():
  #
  if row['proteins'] not in our_test_protein:
    continue #? skip these out-of-ontology
  #
  label_np [row_index] = reorder_array ( row['labels'][col_keep], fix_their_term_to_our )
  label_np_to_keep.append(row['labels'][col_keep])
  prediction_np [row_index] = reorder_array ( row['preds'][col_keep], fix_their_term_to_our )
  prediction_to_keep.append(row['preds'][col_keep])
  # this_label = [ label for label in row['annotations'] if label in label_in_onto ] # ! ! very important to not filter the labels here. not use [col_keep]
  # label_to_keep.append( set( this_label ) )
  label_to_keep.append( row['annotations'] )
  row_index = row_index + 1

#
# remove row that has all zero

output = {'prediction':prediction_np, 'true_label':label_np}

pickle.dump(output, open(output_path,'wb'))

# df['annotations'] = label_to_keep
# df['labels'] = label_np_to_keep
# df['preds'] = prediction_to_keep

# df.to_pickle('predictions_filter_by_mf.pkl')

# df



# {GO:0043296, GO:0030054, GO:0030031, GO:000382...
# labels         [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
