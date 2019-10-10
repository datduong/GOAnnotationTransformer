
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import pickle,re,sys,os

os.chdir('/local/datdb/deepgo/data/BertNotFtAARawSeqGO/fold_1mf2embGeluE768H4L10I768')
attention = pickle.load (open("GO2all_attention_O54992_B3PC73.pickle","rb"))

prot = list ( attention.keys() ) 
print (prot)

for p in prot: 
  for layer in range(10): 
    for head in range (4): 
      matrix = attention[p] [layer] [head]
      np.savetxt ( p + 'layer' + str(layer) + 'head' + str(head)+'.csv', matrix, delimiter=',')




for p in prot: 
  for layer in range(10): 
    for head in range (4): 
      matrix = attention[p] [layer] [head]
      plt.clf()
      plt.imshow(matrix,interpolation='nearest', cmap=cm.inferno)
      # ax.colorbar() # Add a scale bar
      # plt.title(p + ' layer ' + str(layer) + ' head ' + str(head))
      plt.savefig( p + 'layer' + str(layer) + 'head' + str(head)+'.png' )


