import re,sys,os,pickle
import numpy as np
import pandas as pd

#### to weight loss function

import networkx
import obonet

def weighted_by_num_child(graph, node): ##!! only take is_a
  num_child = 1.0 ## avoid division by 0
  for parent, child, key in graph.in_edges(node, keys=True):
    if key == 'is_a':
      num_child = num_child + 1
  return 1.0/num_child

work_dir = '/local/datdb/deepgo/data'
os.chdir (work_dir)
# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True) if 'OBSOLETE' not in data.get('def')}

os.chdir('/local/datdb/deepgo/data/train')

for onto in ['bp','cc','mf']: 
  GOfile = open('deepgo.'+onto+'.csv',"r")
  fout = open('deepgo.'+onto+'.weight.csv',"w")
  for line in GOfile:
    line = line.strip()
    weight = weighted_by_num_child(graph,line)
    fout.write(line+"\t"+str(weight)+'\n')
  ##
  fout.close()
  GOfile.close()


# label_2test_array = pd.read_csv('deepgo.'+onto+'.weight.csv',header=None,sep="\t")
# label_2test_array = label_2test_array.sort_values(by=[0], ascending=True) 
