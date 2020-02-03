


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet

work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)

graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet


## take a list of GO terms, view the def, this is useful for human interpretation 

# list_in ="""
# "GO0000030" "GO0000049" "GO0000149" "GO0000166" "GO0000979" "GO0000980"
# "GO0001076" "GO0001085" "GO0001098" "GO0001102" "GO0001104" "GO0001158"
# "GO0001190" "GO0001191" "GO0001653" "GO0001664" "GO0001786" "GO0001871"
# "GO0001882" "GO0001883" "GO0001948" "GO0002020" "GO0002039" "GO0003678"
# "GO0003682" "GO0003684" "GO0003688" "GO0000287" "GO0000975" "GO0000976"
# "GO0000977" "GO0000978" "GO0000981" "GO0000982" "GO0000987" "GO0000988"
# "GO0000989" "GO0001012" "GO0001046" "GO0001047" "GO0001067" "GO0001071"
# "GO0001077" "GO0001078" "GO0001159" "GO0001227" "GO0001228" "GO0003676"
# "GO0003677" "GO0003690"
# """

list_in ="""
"GO0000030" "GO0000049" "GO0000149" "GO0000975" "GO0000976" "GO0000988"
"GO0000989" "GO0001047" "GO0001067" "GO0001071" "GO0001076" "GO0001085"
"GO0001098" "GO0001664" "GO0001786" "GO0001948" "GO0002020" "GO0003678"
"GO0003682" "GO0003684" "GO0003688" "GO0003690" "GO0000166" "GO0000287"
"GO0001882" "GO0001883" "GO0003676" "GO0003677" "GO0000977" "GO0000978"
"GO0000979" "GO0000980" "GO0000981" "GO0000982" "GO0000987" "GO0001012"
"GO0001046" "GO0001077" "GO0001078" "GO0001102" "GO0001104" "GO0001158"
"GO0001159" "GO0001190" "GO0001191" "GO0001227" "GO0001228" "GO0001653"
"GO0001871" "GO0002039"
"""

list_in = re.sub('"','',list_in)
go_list = list_in.split()

# graph.node['GO:0000002']['def']

for node in go_list: 
  name = re.sub('^GO',"GO:",node) 
  print (name + '\t' + graph.node[name]['def'].strip())



