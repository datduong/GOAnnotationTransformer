
import sys,re,os,pickle
import pandas as pd 

os.chdir("/local/auppunda/auppunda/deepgoplus/deepgoplus.bio2vec.net/data/data-cafa")

df = pd.read_pickle('test_data.pkl')
df [df['proteins']=='T100900015822']

T100900015822


# >>> df [df['proteins']=='T100900015822']                                         
#           proteins                                          sequences annotations
# 305  T100900015822  MTVRNIASICNMGTNASALEKDIGPEQFPINEHYFGLVNFGNTCYC...          {}
