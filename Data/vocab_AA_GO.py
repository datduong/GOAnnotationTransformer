import os,sys,re,pickle

## use AA sequence, and GO only, without definitions ... so maybe we can trim down GPU usage by using only 1 single letter. 

os.chdir("/u/scratch/d/datduong/BERTPretrainedModel/cased_L-12_H-768_A-12AAraw2016")

letters = 'A, E, I, O, U, B, C, D, F, G, H, J, K, L, M, N, P, Q, R, S, T, V, X, Z, W, Y'.split(',')
letters = sorted ( [let.strip() for let in letters] )


fout = open("vocabAA.txt","w") # +GO
fin = open("vocab+GO.txt","r")
for line in fin: 
  line = line.strip()
  if line in '[UNK] [CLS] [SEP] [MASK] [PAD]': 
    fout.write(line+"\n")
  # elif bool(re.findall(r'GO[0-9]+',line)) or bool(re.findall(r'unused[0-9]',line)): 
  #   fout.write(line+"\n")
  else: 
    if (len(line)==1) and (line in letters) :
      fout.write(line+"\n")


fout.close()
fin.close() 
# 2092
