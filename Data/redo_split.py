

import os,re,sys,pickle
import pandas as pd
import numpy as np

## one protein is missing in larger data set ... doesn't matter very much. but why though? 

for onto in ['mf','cc','bp']:
  df1original = pd.read_csv("/u/scratch/d/datduong/deepgo/data/train/fold_1/train-"+onto+".tsv",sep='\t')
  df2original = pd.read_csv("/u/scratch/d/datduong/deepgo/data/train/fold_1/dev-"+onto+".tsv",sep='\t')
  # expand
  df1 = pd.read_csv("/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/train-"+onto+".tsv",sep='\t')
  df2 = pd.read_csv("/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/dev-"+onto+".tsv",sep='\t')
  expand = pd.concat([df1,df2])
  # df.loc[~df['column_name'].isin(some_values)]
  trainExpand = expand.loc [ expand['Entry'].isin(list(df1original['Entry'])) ]
  devExpand = expand.loc [ expand['Entry'].isin(list(df2original['Entry'])) ]
  # write out
  trainExpand.to_csv("/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/train-"+onto+"-same-origin.tsv",sep="\t",index=None)
  devExpand.to_csv("/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/dev-"+onto+"-same-origin.tsv",sep="\t",index=None)


