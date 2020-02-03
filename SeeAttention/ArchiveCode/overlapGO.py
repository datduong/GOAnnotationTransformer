



from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata, string, re, sys, os, pickle, pickle, gzip

import numpy as np
import pandas as pd 

from copy import deepcopy

import networkx
import obonet


## what if we remove parents ?
## do we need to have complete path ? 

work_dir = '/u/flashscratch/d/datduong/deepgo/data/'
os.chdir (work_dir)

# Read the taxrank ontology
graph = obonet.read_obo('go.obo') # https://github.com/dhimmel/obonet

x1 = networkx.descendants(graph, 'GO:0016491')

x2 = networkx.descendants(graph, 'GO:0003876')

x1.intersection(x2)

