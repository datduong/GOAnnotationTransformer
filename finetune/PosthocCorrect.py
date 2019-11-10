

import pickle,sys,os,re
import pandas as pd
import numpy as np
import torch

import networkx
import obonet

## recursively take max of all children to assign prob. to 1 node

def GetDescentNode (graph,label_array):
  label_lookup = { name:index for index,name in enumerate(label_array) }
  children = {}
  for index,name in enumerate(label_array):
    ## @graph takes GO:01234567 format
    to_get = sorted ( [re.sub('GO','GO:',name)] + list ( networkx.ancestors(graph, re.sub('GO','GO:',name) ) ) ) ## add itself, networkx.ancestors returns subterms. https://github.com/dhimmel/obonet
    ## @label_lookup use GO1234567 format. because bert split style
    to_get = [re.sub(r'GO:','GO',n) for n in to_get] ## put back GO:
    children[name] = np.array ( [label_lookup[n] for n in to_get if n in label_lookup] )
  return children, label_lookup


def TakeMax (prob,label_array,children):
  # notice, do not need to do recusrive max, because max(x, max(y,z)) = max(x,y,z)
  # @prob is num_sample x num_label
  # for each row get column of its children, take max and replace.
  for col,name in enumerate(label_array): ## for each observation
    if len(children[name])> 0 :
      prob[ : , col ] = np.max ( prob[ : , children[name] ], 1 ) ## replace the column of this label @name, with the max-over-row of its children
  return prob


def PosthocMax (label_array,prob,go_obo_path='/local/datdb/deepgo/data/go.obo'):
  graph = obonet.read_obo(go_obo_path) ## create graph
  children, label_lookup = GetDescentNode (graph,label_array)
  # print (children['GO0003677'])
  return TakeMax (prob,label_array,children)

