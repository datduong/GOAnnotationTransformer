
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
  # child_of_root = [ re.sub(':','',c) for c in child_of_root ]
  return child_of_root + [root_node[onto]] # [ re.sub(':','',root_node[onto]) ]


#
os.chdir('/u/scratch/d/datduong/UniprotJan2020')

new_dir = '/u/scratch/d/datduong/UniprotJan2020/AddIsA/'
if not os.path.exists(new_dir):
  os.mkdir(new_dir)


num_line = {'bp':353893,'cc':323810,'mf':376207}
root_node = {'bp':'GO:0008150' ,'cc':'GO:0005575' , 'mf':'GO:0003674' }
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet


label_fraction = {}
fin = open(new_dir+"label-fraction-count.tsv","r")
for line in fin:
  line = line.split('\t')
  label_fraction[line[0]] = float(line[1])

#
fin.close()

to_remove = get_child_of_root(root_node,'mf') ##!! should only remove bp ?
to_remove = to_remove + get_child_of_root(root_node,'bp') ##!! should only remove bp ?
to_remove = to_remove + get_child_of_root(root_node,'cc') ##!! should only remove cc ?
#### remove by fraction count
to_remove = [ t for t in to_remove if (t in label_fraction) and (label_fraction[t]>0.2) ]
to_remove2 = [ key for key,value in label_fraction.items() if value>0.2 ]
to_remove = to_remove + ['GO:0008150' , 'GO:0005575' , 'GO:0003674']
to_remove = list ( set ( to_remove + to_remove2 ) )
print (len(to_remove))


for onto in ['cc','mf','bp']:
  labelset = {}
  # uniprot-filter3-cc-bonnie.tsv
  fin = open("uniprot-"+onto+"-isa-bonnie.tsv","r") # name, seq, label, vec, motif
  # Entry Gene ontology IDs Sequence  prot3dvec
  # Q8YW84  GO:0003674;GO:0003824;GO:0005488;GO:0016491;GO:0016651;GO:0016655;GO:0048037;GO:0048038
  for index,line in enumerate(fin):
    if index==0: 
      continue ## skip header
    label = line.split('\t') [1] ## label only
    label = label.split(';')
    for l in label:
      if l in labelset:
        labelset[l] = 1 + labelset[l]
      else:
        labelset[l] = 1
  #
  fin.close()
  # fout = open (new_dir+'/'+onto+'-label.tsv','w')
  fout = open (new_dir+'/'+onto+'-label-rm20.tsv','w')
  label = sorted( list( labelset.keys() ) )
  for l in sorted(label):
    #### the first run get all fraction count, then we reuse the same code to remove labels
    # fout.write(l+'\t'+str( labelset[l]*1.0/num_line[onto] )+'\n')
    if l not in to_remove: ##!! remove labels
      fout.write(l+'\t'+str( labelset[l] )+'\n')
  fout.close()



