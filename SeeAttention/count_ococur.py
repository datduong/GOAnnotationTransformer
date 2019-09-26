
import os,sys,pickle,re
import numpy as np 
import pandas as pd 

os.chdir('/u/scratch/d/datduong/deepgo/data/train/fold_1/TokenClassify')

label_2test_array = pd.read_csv("/u/scratch/d/datduong/deepgo/data/deepgo.mf.csv",header=None,dtype=str)
label_2test_array = sorted ( label_2test_array[0] )
label_2test_array = [ re.sub(":","",g) for g in label_2test_array ] 
label_position_map = { value:index for index,value in enumerate(label_2test_array) }

num_label = len(label_2test_array)

GO2GO = np.zeros((num_label,num_label))

df = pd.read_csv("/u/scratch/d/datduong/deepgo/data/train/fold_1/train-mf.tsv",sep="\t",dtype=str) # Entry Gene ontology IDs Sequence  Prot Emb

for index,row in df.iterrows(): 
  go_list = re.sub(r":","",row['Gene ontology IDs'])
  go_list = sorted( [ g.strip() for g in go_list.split(";") ] )
  for g1 in go_list: 
    for g2 in go_list: 
      GO2GO [ label_position_map[g1] , label_position_map[g2] ] = GO2GO [ label_position_map[g1] , label_position_map[g2] ] + 1



## 
df = pd.DataFrame(GO2GO, columns=label_2test_array, index=label_2test_array)
df.to_csv ('GO2GO_mf_count.csv',index=None,sep=",") ## later in plotting, from col names we can get row names.


