
import os, sys, re, pickle
import numpy as np
import pandas as pd

#### should we keep all the labels ? too many ??

## remove 1 below root 
import networkx
import obonet

def get_child_of_root(root_node,onto):
  child_of_root = []
  for parent, child, key in graph.in_edges(root_node[onto], keys=True):
    if key == 'is_a':
      child_of_root.append(parent)
  child_of_root = [ re.sub(':','',c) for c in child_of_root ]
  return child_of_root + [ re.sub(':','',root_node[onto]) ]

#
os.chdir('/u/scratch/d/datduong/UniprotJan2020')

num_line = {'bp':353893,'cc':323810,'mf':376207}

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
root_node = {'bp':'GO:0008150' ,'cc':'GO:0005575' , 'mf':'GO:0003674' }

# to_remove = get_child_of_root(root_node,'mf') ##!! should only remove bp ?
# to_remove = get_child_of_root(root_node,'bp') ##!! should only remove bp ?
# to_remove = get_child_of_root(root_node,'cc') ##!! should only remove cc ?



for onto in ['cc','mf','bp']:
  labelset = {}
  # uniprot-filter3-cc-bonnie.tsv
  fin = open("uniprot-filter3-"+onto+"-bonnie.tsv","r") # name, seq, label, vec, motif
  for index,line in enumerate(fin):
    label = line.split('\t') [2] ## label only
    label = label.split()
    for l in label:
      if l in labelset:
        labelset[l] = 1 + labelset[l]
      else:
        labelset[l] = 1
  #
  fin.close()
  fout = open (onto+'-label.tsv','w')
  label = sorted( list( labelset.keys() ) )
  for l in sorted(label):
    # if l not in to_remove: 
    fout.write(l+'\t'+str( labelset[l]*1.0/num_line[onto] )+'\n')
  fout.close()




