
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


import networkx
import obonet


# work_dir = '/u/flashscratch/d/datduong/goAndGeneAnnotationMar2017/'
work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)
# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
len(graph) # Number of nodes
graph.number_of_edges() # Number of edges
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

fout = open("/u/flashscratch/d/datduong/Onto2Vec/BertQnliGOEmb768/full.txt","w")
fin = open("/u/flashscratch/d/datduong/Onto2Vec/GOVectorData/2016DeepGOData/AllAxioms_2016.lst","r")
counter = 0 
for line in fin: 
  line = re.sub("GO_","GO:",line)
  go_array = re.findall("GO:[0-9]{7,}",line) 
  if len(go_array)<2: 
    continue
  ## need to get left/right as entailment. 
  fout.write(str(counter)+"\t"+go_array[0]+"\t"+go_array[1::]+"\tentailment\n")
  ## sample one random
  while True: 
    one_random = np.random.choice(list(id_to_name.keys()), size=1)
    if (one_random not in networkx.descendants(graph, go_array[0])) and (one_random not in networkx.ancestors(graph, go_array[0])) : 
      break 
  # 
  counter = counter + 1 
  fout.write(str(counter)+"\t"+go_array[0]+"\t"+one_random+"\tnot_entailment\n")
  counter = counter + 1 


