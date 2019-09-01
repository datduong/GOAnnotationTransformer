
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


# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/'
work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
len(graph) # Number of nodes
graph.number_of_edges() # Number of edges

networkx.is_directed_acyclic_graph(graph) # Check if the ontology is a DAG

# Mapping from term ID to name
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')} ## by default obsolete already removed
# id_to_name['GO:0000002']

# Find all superterms of species. Note that networkx.descendants gets
# superterms, while networkx.ancestors returns subterms.
# networkx.descendants(graph, 'GO:0000002') ## return parents https://www.ebi.ac.uk/QuickGO/term/GO:0000002 GO:1904426

# the first list contains the index of the source nodes,
# while the index of target nodes is specified in the second list.
# edge_index = torch.tensor([[0, 1, 2, 0, 3],
#                            [1, 0, 1, 3, 2]], dtype=torch.long)



fout = open("GO_branch_split_half.txt","w")

##!! Each node must appear at least once so we can encode it

for node_name in tqdm(id_to_name): ## for each node

  path_to_root = []
  node_to_look = [node_name]

  while len(node_to_look) > 0 :

    graph_out = graph.out_edges(node_name, keys=True)
    if len(graph_out) == 0: ## exit
      break

    path_to_root.append(node_name) ## immediately append before look up parents 
    all_parents = [] 
    for child, parent, key in graph_out:
      # @key tells "is_a" or "part_of" ...
      if key == 'is_a':
        all_parents.append(parent)

    ## the parents of this node
    next_node = np.random.choice(all_parents,size=1)[0] ## move up to parent node, pick 1 at random
    node_to_look[0] = next_node
    node_name = next_node ## move to parent, go up the tree

  ##
  ## split branch in half 
  num_path = len ( path_to_root ) 
  if len(path_to_root) > 1: 
    path1 = path_to_root [ 0: (num_path//2) ] 
    path2 = path_to_root [ (num_path//2) :: ] 
    fout.write ( re.sub(":","", " ".join(str(p) for p in path1) ) + "\n" + re.sub(":","", " ".join(str(p) for p in path2) ) + "\n\n" )


fout.close() 


