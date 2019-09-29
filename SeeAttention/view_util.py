

import sys,re,os,pickle
import numpy as np
import pandas as pd

from pytorch_transformers import BertTokenizer


def normalize_to_sum_1 (a):
  return a/a.sum(axis=1,keepdims=1)

def get_word_index_in_array (vocab_dict,some_array):
  # return [ vocab_dict[lab] for lab in sorted(some_array) ]
  # @vocab_dict can be object BERT tokenizer
  return [ vocab_dict._convert_token_to_id(lab) for lab in sorted(some_array) ]


def get_att_weight (matrix,input_ids,label_names_in_index,seq_in_index=None):
  # @input_ids is word indexing, it will have CLS and SEP
  # extract where labels are on the row
  # extract where sequences are on the columns.
  # @matrix is attention weight, where row sums to 1. so, we see how much col j contributes to row i.
  # can't label np array ? https://stackoverflow.com/questions/44708911/structured-2d-numpy-array-setting-column-and-row-names
  # much easier if we can pull stuffs by row and col names
  # df = pd.DataFrame(matrix, columns=input_ids, index=input_ids)
  # https://thispointer.com/select-rows-columns-by-name-or-index-in-dataframe-using-loc-iloc-python-pandas/
  # https://stackoverflow.com/questions/19161512/numpy-extract-submatrix

  ## must be really careful here. @label_names_in_index is GO1-->WordIndex1 ... indices = np.argwhere(np.in1d(a,b))

  ## get which @input_ids positions are the GO names, so we extract only these positions
  go_position = np.argwhere(np.in1d(input_ids,label_names_in_index)).transpose()[0]  ## get back which positions are GO names, use .transpose()[0] to get back 1D np array
  GO2GO = matrix[ np.ix_ ( go_position,go_position ) ] ## extract submatrix that has only GO-vs-GO attention 
  out = ( normalize_to_sum_1(GO2GO) , ) ## tuple

  if seq_in_index is not None:
    aa_position = np.argwhere(np.in1d(input_ids,seq_in_index)).transpose()[0]
    ## add into tuple so use (something, )
    out = out + ( normalize_to_sum_1 ( matrix[ np.ix_ ( go_position,aa_position ) ] ) , )  ## row sum to 1, so we see which segment contribute most to this GO 
  return out


def get_best_prob (matrix,top_k=20) :
  ## return top k hit of col j on the given row i. remember, row adds to 1. so it is how much j contributes to i
  position = np.argsort(matrix, axis=1)[:,::-1] ## use -1 to reverse ordering from largest to smallest
  return position [:,0:top_k] # return only top_k best j that contributes most to i

# def get_best_prob_quantile (matrix,quantile=.95) :
#   ## return top k hit of col j on the given row i. remember, row adds to 1. so it is how much j contributes to i
#   cut_off = np.quantile(matrix,axis=1,q=[quantile]) ## get cutoff one segment wrt. many GO
#   cut_off = np.reshape ( cut_off.repeat(matrix.shape[0]), matrix.shape ) ## repeat same number of GO, reshape into same shape as @matrix 
#   pass 

from sympy import Interval, Union # https://stackoverflow.com/questions/15273693/python-union-of-multiple-ranges
def union(data):
  """ Union of a list of intervals e.g. [(1,2),(3,4)] """
  intervals = [Interval(begin, end) for (begin, end) in data]
  u = Union(*intervals)
  return [list(u.args[:2])] if isinstance(u, Interval) \
    else list(u.args)


def get_best_range (matrix, go_names, expand=5, max_bound=0):
  # take best position in @matrix, then expand left right by @expand, next we union the ranges
  best_range = {}
  for i in range(matrix.shape[0]): ## for each GO term, we get the best segment 
    up_range = matrix[i] + expand ## matrix operation, so we compute for all "best" locations for one GO term, in 1 single row
    low_range = matrix[i] - expand
    
    up_range [up_range>max_bound] = max_bound ## must bound these values by AA length
    low_range [low_range<0] = 0

    this_range = union ( [ r for r in zip(low_range,up_range) ] )
    best_range[go_names[i]] = this_range ## note that @go_names must match exact ordering in the vocab.txt
  return best_range


def return_best_segment (attention, tokenizer, sequence, go_names, expand, top_k, max_bound ): ## @attention is some weight of seq-vs-GO  
  ## how do we average or something ?? for 2 given sequences, and the same GO terms ?? 
  
  output = {} 

  where_best = get_best_prob ( attention, top_k ) # @where_best is matrix num_sequence x num_GO 

  best_range = get_best_range(where_best,go_names,expand, max_bound)

  ## get the actual sequence ?? 
  for go in best_range : # @best_range = {go1:RangeList1, go2:RangeList2}
    best_segment = [] 
    for start_end in best_range[go]: 

      # @start_end is object @Interval, or it is a @list 

      if isinstance(start_end, list):
        seg = [ tokenizer._convert_id_to_token (s) for s in sequence[ start_end[0] : start_end[1] ] ] 
      else: 
        seg = [ tokenizer._convert_id_to_token (s) for s in sequence[ start_end.left : start_end.right ] ] 

      best_segment.append ( "".join(seg) ) ## convert [(1,10) (30-40)] --> [ ABCXYZ , ]

    ## best segment of this sequence to this go term 
    output[go] = best_segment

  return output ## this is for 1 sequence and many GO term. AT ONE ATTENTION HEAD. 


def get_best_many_head (last_layer_att, tokenizer, sequence, num_head, go_names, expand, top_k) : 
  ## different sequence has their own unique length. so we can't do batch model or broadcast style. 
  ## loop over each sample, for each sample, then loop over each head. ?? 
  # @last_layer_att will be #obs x #head x #word x #word, so we can pass in last_layer_att=last_layer_att[0] or something. 
  head = {}
  for head_id in range(num_head) : 
    head[head_id] = return_best_segment ( last_layer_att[head_id], tokenizer, sequence, go_names, expand, top_k ) 
  return head 










