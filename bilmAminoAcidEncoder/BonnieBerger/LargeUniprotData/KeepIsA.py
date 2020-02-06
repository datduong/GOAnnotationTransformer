

import os, sys, re, pickle
import numpy as np
import pandas as pd

import networkx
import obonet


#### because obonet calls all ancestors, we have to keep only is-a relationships

os.chdir('/u/scratch/d/datduong/UniprotJan2020')

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet
root_node = {'bp':'GO:0008150' ,'cc':'GO:0005575' , 'mf':'GO:0003674' }


def get_parent(node):
  parent_array = []
  for child, parent, key in graph.out_edges(node, keys=True):
    if key == 'is_a':
      parent_array.append(parent)
  return parent_array


def find_lowest_node(labels):
  # labels = set ( [re.sub('GO','GO:',l) for l in labels] )
  labels = set(labels)
  lowest = None
  for l in labels:
    ancestors = set( networkx.descendants(graph, l) ) ## weird naming convention
    ##!! check labels intersect with ancestor should be 0 for lowest level node
    print ( len( labels - ancestors ) )
    ##!! does not work properly
    if len( labels - ancestors ) == 1: ## has all ancestors ????
      lowest = l
      break
  return lowest ## should we print something ?


def AddIsA (labels):
  ## @labels is one single string like GO0042651, or GO:0005737
  #### keep is-a only in an array of labels
  # labels = re.sub('GO','GO:',labels)
  # labels = labels.split()
  # lowest_node = find_lowest_node ( labels )
  # print (lowest_node)
  to_get_next = [labels]
  new_labels = [labels] ## all parents of l with is-a
  #### breadth first search style.
  while len(to_get_next)>0:
    ## break point is when @to_get_next == 0
    for l in to_get_next:
      to_get_next = to_get_next + list ( get_parent(l) )
      to_get_next.remove(l) ## remove this node @l so we move past it
    ## add to new label
    new_labels = new_labels + to_get_next ## add all the "next layer node"
  #
  new_labels = sorted( list (set (new_labels)) )
  return new_labels


def AddIsA2array (labels_string):
  new_labels = []
  labels_string = labels_string.split(';') # GO:0030430;GO:0044163
  for label in labels_string:
    new_labels = new_labels + AddIsA(label)
  return sorted( list (set (new_labels)) )


# AddIsA2array('GO:0006183;GO:0006228;GO:0006241')
# AddIsA2array('GO:0042651')

# Entry Gene ontology IDs Sequence  prot3dvec
# uniprot-cc-bonnie.tsv


# onto = 'mf'
for onto in ['cc','bp']:

  fin = open ('uniprot-'+onto+'-bonnie.tsv','r')
  fout = open ('uniprot-'+onto+'-isa-bonnie.tsv','w')
  for index,line in enumerate(fin):
    if index == 0: ## header
      fout.write(line)
      continue
    line = line.strip().split('\t')
    annot = AddIsA2array( line[1] )
    annot = ';'.join(annot)
    fout.write(line[0]+"\t"+annot+"\t"+line[2]+"\t"+line[3]+"\n")

  ##
  fout.close()
  fin.close()





