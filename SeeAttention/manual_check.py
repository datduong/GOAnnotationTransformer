import pickle,os,sys,re
import numpy as np
import pandas as pd

GO2AA = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf/GO2AA_attention.pickle","rb"))
GO2AA_quantile = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf/GO2AA_attention_quantile.pickle","rb"))

label_2test_array = sorted ( ['GO0002039','GO0000287','GO0000049'] ) ## add more later

GOname = 'GO0002039' # "GO0002039" GO0000049 GO0000287
## get only uniprot number needed ??
AAwithThisGO = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/"+GOname+"seq.txt",sep="\t",dtype=str,header=None)

train_data = pd.read_csv( "/local/datdb/deepgo/data/train/fold_1/train-mf.tsv",sep="\t",dtype=str)
all_name = list(train_data['Entry'])

which_index_number = {a : all_name.index(a) for a in list(AAwithThisGO[0]) }

# {'Q6PCD5': 830, 'O54992': 1340, 'Q8IW41': 2097, 'Q96S44': 2799, 'Q9XVN3': 5441, 'Q8N726': 7318, 'Q9NQR1': 7565, 'Q9NXV6': 7657, 'Q00987': 8126, 'P53801': 8460, 'P46580': 8472, 'Q9BV47': 9884, 'Q969K3': 9945, 'Q8WZ73': 10028, 'O60341': 11042, 'Q9H4B4': 11565, 'Q5VTR2': 12337, 'Q16288': 12633, 'O35625': 12693, 'O60285': 14278, 'P49615': 14896, 'Q15327': 14992, 'P49841': 16499, 'P05197': 17111, 'P28360': 17607, 'P17679': 18752, 'Q95SP2': 18761}

# use index 1340
index = 1
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


O54992
GO0002039
head 0
seg KETSILEEYSINWT prob 0.0023890086449682713
seg TDKVDRLKLAEVVK prob 0.002174914116039872
head 1
seg ASQVTKQIALALQH prob 0.002170447027310729
seg LFKDNSLDAPVKLC prob 0.0022153756581246853
head 2
seg GFAKVDQGDLMTPQ prob 0.0024300063960254192
seg WSQISEMAKDVVRK prob 0.002282105851918459
head 3
seg LDAPVKLCDFGFAK prob 0.005734909325838089
seg EQLANMRIQDLKVS prob 0.008077656850218773
http://elm.eu.org/elms/DEG_Kelch_Keap1_2
p-value 	1.71e-03
E-value 	2.80e-01
q-value 	2.78e-01
head 4
seg IIEVFANSVQFPHE prob 0.0022445127833634615
seg PTSPTPYTYNKSCD prob 0.0021923198364675045
head 5
seg FTEKQASQVTKQIA prob 0.002303825691342354
seg NIAHRDLKPENLLF prob 0.0022610228043049574


GO0000287
head 0
seg YSKHHSRTIPKDMR prob 0.003230223897844553
seg AEQLANMRIQDLKV prob 0.002999737625941634
head 1
seg RHQKEKSGIIPTSP prob 0.004756825510412455
seg AEQLANMRIQDLKV prob 0.004767631646245718
head 2
seg SQHRHFTEKQASQV prob 0.0024723331443965435
seg NNPILRKRKLLGTK prob 0.002081793500110507
head 3
seg KETSILEEYSINWT prob 0.005201298277825117
seg RVCVKKSTQERFAL prob 0.005859438795596361
http://elm.eu.org/elms/LIG_Dynein_DLC8_1
p-value 	2.28e-03
E-value 	3.73e-01
q-value 	2.03e-01
head 4
seg PFYSKHHSRTIPKD prob 0.0029803626239299774
seg LKLAEVVKQVIEEQ prob 0.0023021905217319727
head 5
seg QFTPYYVAPQVLEA prob 0.0024008911568671465
seg GIIPTSPTPYTYNK prob 0.0021255887113511562


GO0000049
head 0
seg EMMEGGELFHRISQ prob 0.0025375448167324066
seg AEQLANMRIQDLKV prob 0.0026187708135694265
head 1
seg RHQKEKSGIIPTSP prob 0.009229096584022045
seg AEQLANMRIQDLKV prob 0.010564419440925121
http://elm.eu.org/elms/DEG_Kelch_Keap1_2
p-value 	1.72e-03
E-value 	2.82e-01
q-value 	2.80e-01
head 2
seg GFAKVDQGDLMTPQFTPYY prob 0.0025684728752821684
head 3
seg ESSPRARLLIVMEM prob 0.006918582133948803
seg EQLANMRIQDLKVS prob 0.008738559670746326
http://elm.eu.org/elms/DEG_Kelch_Keap1_2
p-value 	1.71e-03
E-value 	2.80e-01
q-value 	2.78e-01
head 4
seg MEGGELFHRISQHR prob 0.0031595833133906126
seg LLKVKPEERLTIEG prob 0.0023989847395569086
head 5
seg MSEDSD prob 0.003367926925420761
seg NIAHRDLKPENLLF prob 0.0026488006114959717


