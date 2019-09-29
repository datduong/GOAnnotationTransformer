import pickle,os,sys,re
import numpy as np
import pandas as pd 

GO2AA = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf/GO2AA_attention.pickle","rb"))

GOname = "GO0002039" 

## get only uniprot number needed ?? 
AAwithThisGO = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/GO0002039seq.txt",sep="\t",dtype=str,header=None) 

train_data = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/train-mf.tsv",sep="\t",dtype=str) 
all_name = list(train_data['Entry'])

which_index_number = {a : all_name.index(a) for a in list(AAwithThisGO[0]) }

# {'Q6PCD5': 830, 'O54992': 1340, 'Q8IW41': 2097, 'Q96S44': 2799, 'Q9XVN3': 5441, 'Q8N726': 7318, 'Q9NQR1': 7565, 'Q9NXV6': 7657, 'Q00987': 8126, 'P53801': 8460, 'P46580': 8472, 'Q9BV47': 9884, 'Q969K3': 9945, 'Q8WZ73': 10028, 'O60341': 11042, 'Q9H4B4': 11565, 'Q5VTR2': 12337, 'Q16288': 12633, 'O35625': 12693, 'O60285': 14278, 'P49615': 14896, 'Q15327': 14992, 'P49841': 16499, 'P05197': 17111, 'P28360': 17607, 'P17679': 18752, 'Q95SP2': 18761}



for head in range(6):
  print ( GO2AA[1340][head][GOname] )

['KETSILEEYSINWT', 'TDKVDRLKLAEVVK']
http://elm.eu.org/elms/MOD_PK_1 http://elm.eu.org/elms/DOC_GSK3_Axin_1
['ASQVTKQIALALQH', 'LFKDNSLDAPVKLC']
http://elm.eu.org/elms/DEG_Kelch_KLHL3_1 http://elm.eu.org/elms/DOC_PP2B_LxvP_1
['GFAKVDQGDLMTPQ', 'WSQISEMAKDVVRK']
http://elm.eu.org/elms/LIG_RGD http://elm.eu.org/elms/DEG_SCF_COI1_1
['LDAPVKLCDFGFAK', 'EQLANMRIQDLKVS']
http://elm.eu.org/elms/DOC_PP2B_LxvP_1 http://elm.eu.org/elms/DEG_Kelch_Keap1_2
['IIEVFANSVQFPHE', 'PTSPTPYTYNKSCD']
http://elm.eu.org/elms/LIG_SPRY_1 http://elm.eu.org/elms/MOD_TYR_DYR
['FTEKQASQVTKQIA', 'NIAHRDLKPENLLF']
http://elm.eu.org/elms/DEG_Kelch_KLHL3_1 http://elm.eu.org/elms/DOC_MAPK_JIP1_4

