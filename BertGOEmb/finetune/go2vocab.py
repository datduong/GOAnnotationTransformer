
import re,pickle,os,sys


fout = open ("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12GO2017/vocab.txt","w")
fout.write('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n') ## padding is index 0 

fin = open("/u/scratch/d/datduong/goAndGeneAnnotationMar2017/go_name_in_obo.csv",'r')
for line in fin: 
  fout.write(re.sub(":","",line))


fin.close()
fout.close() 

