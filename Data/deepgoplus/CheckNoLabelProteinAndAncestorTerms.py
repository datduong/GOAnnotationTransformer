
import sys,re,os,pickle
import pandas as pd

os.chdir("/local/auppunda/auppunda/deepgoplus/deepgoplus.bio2vec.net/data/data-cafa")

df = pd.read_pickle('test_data.pkl')
df [df['proteins']=='T100900015822']

T100900015822


# >>> df [df['proteins']=='T100900015822']
#           proteins                                          sequences annotations
# 305  T100900015822  MTVRNIASICNMGTNASALEKDIGPEQFPINEHYFGLVNFGNTCYC...          {}

# ! check that their test file make sense. do they contain all terms on same path??


import pickle,os,sys,re
import pandas as pd
import numpy as np
from copy import deepcopy
import networkx
import obonet

# read in this same go.obo as they did
graph = obonet.read_obo('/u/scratch/d/datduong/deepgoplus/data-cafa/go.obo') # https://github.com/dhimmel/obonet # graph.node['GO:0003824']['namespace']

terms = 'GO0000975 GO0000976 GO0001067 GO0001071 GO0003674 GO0003676 GO0003677 GO0003690 GO0003700 GO0005488 GO0030170 GO0043167 GO0043168 GO0043565 GO0044212 GO0048037 GO0097159 GO1901363 GO1990837'.split()
terms = sorted ( [re.sub('GO','GO:',t) for t in terms] )

terms_to_use = {}
for node in terms:
  for child, parent, key in graph.out_edges(node, keys=True):
    if key == 'is_a':
      if parent not in terms_to_use:
        terms_to_use[parent] = 1
      else:
        terms_to_use[parent] = 1 + terms_to_use[parent]

#
terms_to_use

terms_to_use = sorted (list ( terms_to_use.keys() ) )
terms = sorted (terms)

set(terms_to_use) - set(terms) ## have more ancestors than what is shown in their file
set(terms) - set(terms_to_use) ## these are leaf nodes

