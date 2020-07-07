import sys,re,os,pickle
import numpy as np
import pandas as pd



####

## remove 1 below root
import networkx
import obonet


roots = ['GO:0008150','GO:0003674','GO:0005575']

def get_parents (node,graph):
  parent_return = []
  for child, parent, key in graph.out_edges(node, keys=True):
    if (key == 'is_a') and (parent not in roots):
      parent_return.append(parent)
  return parent_return


def calculate_ic(label_counter,graph): # @annot is a set {GO:123, GO:234 ...}
  #! get IC score based on train/test data
  ic = {} # return ic score
  for go_id, n in label_counter.items():
    parents = get_parents(go_id,graph) ## get parents
    if len(parents) == 0:
      min_n = n # no parent, set min as its own occur.
    else:
      min_n = min([label_counter[x] for x in parents])
    ic[go_id] = np.log2(min_n / n) ##!! why base 2 ?? we just go with deepgoplus
  #
  return ic


def count_label (list_file): #! count all the labels in the entire dataset, like deepgoplus
  label_counter = {}
  for fin in list_file:
    fin = pd.read_csv(fin,sep='\t',header=None)
    for index, row in fin.iterrows():
      labels = re.sub ( 'GO','GO:',row[2] )
      labels = labels.strip().split()
      for lab in labels :
        if lab in label_counter:
          label_counter[lab] = label_counter[lab] + 1
        else:
          label_counter[lab] = 1
  #
  return label_counter



graph = obonet.read_obo('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/go.obo') # https://github.com/dhimmel/obonet

os.chdir ('/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000')

list_file = 'deepgoplus.cafa3.test-bonnie-mf.tsv deepgoplus.cafa3.test-bonnie-cc.tsv deepgoplus.cafa3.test-bonnie-bp.tsv deepgoplus.cafa3.train-bonnie-mf.tsv deepgoplus.cafa3.train-bonnie-cc.tsv deepgoplus.cafa3.train-bonnie-bp.tsv'.split()

label_counter = count_label(list_file)
ic = calculate_ic(label_counter,graph)

z = re.sub ( 'GO','GO:', 'GO0032879 GO0032940 GO0044699 GO0044700 GO0044763').split()
for i in z:
  print ( '{} \t {}'.format( label_counter[i], ic[i] ) )


pickle.dump(ic,open('IC-from-all-data.pickle','wb'))


GO0044699
