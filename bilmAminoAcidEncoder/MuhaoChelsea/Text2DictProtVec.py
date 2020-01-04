
import pickle,sys,os,re
import numpy as np
import pandas as pd


#### convert into a dictionary so that we can replace with existing file

os.chdir('/local/datdb/deepgo/data/MuhaoProteinVec')
ProtVecDict = {}
fin = open("protein_embeddings_seqppi.tsv",'r')
for index,line in enumerate(fin): 
  if index==0: 
    continue 
  line = line.split()
  if line[0] in ProtVecDict: 
    print ( 'see duplicated ? {}'.format(line[0]) ) 
  ProtVecDict[line[0]] = ";".join(str(s) for s in line[1::]) 


##!!
print ('\n\ntotal prot {}'.format(len(ProtVecDict)))
pickle.dump(ProtVecDict,open('ProtVecDict.pickle','wb'))


