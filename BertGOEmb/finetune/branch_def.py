

## get def in branch
# GO1 def
# GO2 def (GO2 is parent of GO1) etc ...and

## make data for BERT fine tune.
## take one branch as a "document"

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

from tqdm import tqdm

import pickle, gzip, os, sys, re
import random
from random import shuffle
import numpy as np
import pandas as pd


## extend BERT vocab to all GO terms.
## need a lot of fine tune ??

## use many GO branch to fine tune.
## split branch in half.

import networkx
import obonet

work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/'
# work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
len(graph) # Number of nodes
graph.number_of_edges() # Number of edges

networkx.is_directed_acyclic_graph(graph) # Check if the ontology is a DAG

# Mapping from term ID to name
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')} ## by default obsolete already removed

go_def = {} 
Def = pd.read_csv("go_def_in_obo.tsv",dtype=str,sep="\t")
for index,row in Def.iterrows(): 
  go_def [ row['name'] ] = row['def']


random.seed(a=2019)

fout = open("BERT_go_branch.txt","w")

for node_name in tqdm(id_to_name): ## for each node

  d1 = re.sub(r"\t"," ",go_def[node_name])
  d1 = re.sub(r"\n"," ",d1)
  sent = re.sub(":","",node_name) + " " + d1.lower()

  stack = [node_name] ## start with this leaf
  while len(stack) > 0:

    graph_out = graph.out_edges(node_name, keys=True)
    if len(graph_out) == 0: ## exit
      break

    all_parents = [] 
    for child, parent, key in graph_out:
      # @key tells "is_a" or "part_of" ...
      if key == 'is_a':
        all_parents.append(parent)

    ## the parents of this node
    parent = np.random.choice(all_parents,size=1)[0] ## move up to parent node, pick 1 at random
    node_name = parent ## move to parent, go up the tree

    d2 = re.sub(r"\t"," ",go_def[parent])
    d2 = re.sub(r"\n"," ",d2)

    sent = sent + "\n" + re.sub(":","",parent) + " " + d2.lower() ## one sent per line
    stack[0] = parent ## prepare to get parent of the current parent node

  ## next node, so move to next document 
  fout.write( re.sub("GO:","GO",sent) + "\n\n" ) 


fout.close()

# GO0044711 GO0009058 GO0008152

