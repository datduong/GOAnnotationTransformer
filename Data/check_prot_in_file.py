
import os,re,sys,pickle
import pandas as pd
import numpy as np

## one protein is missing in larger data set ... doesn't matter very much. but why though? 

df1 = pd.read_csv("/local/datdb/deepgo/data/train/fold_1/test-mf.tsv",sep='\t')
df2 = pd.read_csv("/local/datdb/deepgo/dataExpandGoSet/train/fold_1/test-mf.tsv",sep='\t')

set1 = set ( list ( df1['Entry'] ) )
set2 = set ( list ( df2['Entry'] ) )

set1-set2
