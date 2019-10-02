import pickle,os,sys,re
import numpy as np
import pandas as pd

GO2AA = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf_relu/GO2AA_attention.pickle","rb"))
GO2AA_quantile = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf_relu/GO2AA_attention_quantile.pickle","rb"))

label_2test_array = sorted ( ['GO0002039','GO0000287','GO0000049'] ) ## add more later

GOname = 'GO0000049' # "GO0002039" GO0000049 GO0000287
## get only uniprot number needed ??
AAwithThisGO = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/"+GOname+"seq.txt",sep="\t",dtype=str,header=None)

train_data = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/train-mf.tsv",sep="\t",dtype=str)
all_name = list(train_data['Entry'])

which_index_number = {a : all_name.index(a) for a in list(AAwithThisGO[0]) }

# {'Q6PCD5': 830, 'O54992': 1340, 'Q8IW41': 2097, 'Q96S44': 2799, 'Q9XVN3': 5441, 'Q8N726': 7318, 'Q9NQR1': 7565, 'Q9NXV6': 7657, 'Q00987': 8126, 'P53801': 8460, 'P46580': 8472, 'Q9BV47': 9884, 'Q969K3': 9945, 'Q8WZ73': 10028, 'O60341': 11042, 'Q9H4B4': 11565, 'Q5VTR2': 12337, 'Q16288': 12633, 'O35625': 12693, 'O60285': 14278, 'P49615': 14896, 'Q15327': 14992, 'P49841': 16499, 'P05197': 17111, 'P28360': 17607, 'P17679': 18752, 'Q95SP2': 18761}

# use index 1340
index = 830
for head in range(6): # range(6)
  print ('head {}'.format(head))
  for top in range(len(GO2AA[index][head][GOname])):
    print ( 'seg {} prob {}'.format( GO2AA[index][head][GOname][top][0] , np.mean(GO2AA[index][head][GOname][top][1]) ) )


#

for counter,go in enumerate(label_2test_array): 
  print ("\ngo {}".format(go))
  for head in range(6):
    print ('head {}'.format(head))
    print (np.round(GO2AA_quantile[index][head][counter],6))
    print ( max ( GO2AA_quantile[index][head][counter] ) / min (GO2AA_quantile[index][head][counter]) ) 


## we can check what is consider "high prob" for each head
for head in range(6): # range(6)
  all_mean = []
  for index in np.arange(0,len(GO2AA)):
    for top in range(len(GO2AA[index][head][GOname])):
      # print ( 'seg {} prob {}'.format( GO2AA[index][head][GOname][top][0] , np.mean(GO2AA[index][head][GOname][top][1]) ) )
      all_mean.append( np.mean(GO2AA[index][head][GOname][top][1]) )
  ###
  print ('head {} quantile {}'.format(head,np.quantile(all_mean,q=[.25,.5,.75,.9])))




