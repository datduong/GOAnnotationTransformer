


import os,sys,re,pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

#### get some data like zinc fingers etc..

path = '/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess1025/bonnie+motif'
os.chdir(path)

annot_name_arr = ['_prot_annot_type'] # , '_prot_annot_type_topo'

for annot_name in annot_name_arr: 
  for onto_type in ['mf']:
    train = pickle.load(open('train_'+onto_type+annot_name+'.pickle','rb'))
    train_count = {}
    ## must select something to be 'UNK' so that we can handle unseen domain
    fout = open('train_'+onto_type+annot_name+'.txt','w')
    fout.write('Type\tCount\tUnkChance\n')
    for key in sorted (train.keys()) :  # key, value in sorted(train.items(), key=lambda kv: kv[1], reverse=True)
      value = train[key]
      chance = value // 10
      if chance > 100:
        chance = 100
      fout.write(key + "\t"+ str(value) + "\t" + str(chance)+'\n')
      train_count[key] = [value,chance]
    #
    fout.close()
    pickle.dump(train_count,open('train_'+onto_type+annot_name+'_count.pickle','wb'))


# z = nn.Embedding(4,5)
# r = torch.LongTensor(np.array([ [[1,1,0],[2,1,3]], [[0,1,0],[3,0,3]] ] ))
