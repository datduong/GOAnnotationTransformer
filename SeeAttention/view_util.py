

import sys,re,os,pickle
import numpy as np
import pandas as pd


def get_label_word_index (vocab_dict,label_names):
  return [ vocab_dict[lab] for lab in sorted(label_names) ]


def get_label_att (matrix,input_ids,label_names_in_index,seq_in_index=None):
  # @input_ids is word indexing, it will have CLS and SEP
  # extract where labels are on the row
  # extract where sequences are on the columns.
  # @matrix is attention weight, where row sums to 1. so, we see how much col j contributes to row i.
  # can't label np array ? https://stackoverflow.com/questions/44708911/structured-2d-numpy-array-setting-column-and-row-names
  # much easier if we can pull stuffs by row and col names
  df = pd.DataFrame(matrix, columns=input_ids, index=input_ids)
  # https://thispointer.com/select-rows-columns-by-name-or-index-in-dataframe-using-loc-iloc-python-pandas/
  GO2GO = df.loc[label_names_in_index,label_names_in_index]
  out = (GO2GO, ) ## tuple
  if seq_in_index is not None:
    out = out + df.loc[seq_in_index,seq_in_index]
  return out 


def get_best_prob (matrix,top_k=20) : 
  ## return top k hit of col j on the given row i. remember, row adds to 1. so it is how much j contributes to i 
  position = np.argsort(matrix, axis=1)[:,::-1] ## use -1 to reverse ordering from largest to smallest 
  return position [:,0:top_k] # return only top_k best j that contributes most to i


from sympy import Interval, Union # https://stackoverflow.com/questions/15273693/python-union-of-multiple-ranges
def union(data):
  """ Union of a list of intervals e.g. [(1,2),(3,4)] """
  intervals = [Interval(begin, end) for (begin, end) in data]
  u = Union(*intervals)
  return [list(u.args[:2])] if isinstance(u, Interval) \
    else list(u.args)

def get_best_range (matrix,expand=10): 
  # take best position in @matrix, then expand left right by @expand, next we union the ranges 
  best_range = {} 
  for i in matrix.shape[0]: 
    up_range = matrix[i] + expand 
    low_range = matrix[i] -expand 
    this_range = union ( [ r for r in zip(low_range,up_range) ] ) 
    best_range[i] = this_range
  return best_range





