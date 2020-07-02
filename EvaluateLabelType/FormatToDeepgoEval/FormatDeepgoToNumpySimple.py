
import re,sys,os,pickle
import pandas as pd
import numpy as np
## assuming no label ordering,
## we can concat the rows into matrix

def make_1hot (onehot_array,true_label,label_array):
  where = [label_array.index(label) for label in true_label]
  onehot_array [where] = 1
  return onehot_array


os.chdir('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/DeepgoplusSingleModel/')

output_path = 'predictions.mf.numpy'
onto_name = 'mf'

#! need to reorder
our_test_label = pd.read_csv('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'+onto_name+'.tsv',header=None,sep='\t') ##!! do not need to reorder, we already sorted by names
our_test_label = list (our_test_label[0]) ## panda format, take first col
num_label = len(our_test_label)

num_label = len(our_test_label)
df = pd.read_pickle('predictions.mf.pkl')
num_protein = df.shape[0]

label_np = np.zeros( (num_protein,num_label) ) #! 677 mf labels - root
prediction_np = np.zeros( (num_protein,num_label) ) #! 677 mf labels - root


for index,row in df.iterrows():
  label_np [index] = make_1hot ( row['labels'], row['annotations'], our_test_label ) 
  prediction_np [index] = row['preds']


#
# remove row that has all zero

output = {'prediction':prediction_np, 'true_label':label_np}

pickle.dump(output, open(output_path,'wb'))

