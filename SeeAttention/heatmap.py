
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import pickle,re,sys,os

os.chdir('/local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiMutGeluE768H1L12I768PreLabDrop0.1')
attention = pickle.load (open("GO2all_attention_O54992_P23109_P9WNC3.pickle","rb")) # P23109

prot = list ( attention.keys() ) 
print (prot)

for p in prot: 
  for layer in range(12): 
    for head in range (1): 
      matrix = attention[p] [layer] [head]
      np.savetxt ( p + 'layer' + str(layer) + 'head' + str(head)+'.csv', matrix, delimiter=',')




# for p in prot: 
#   for layer in range(10): 
#     for head in range (4): 
#       matrix = attention[p] [layer] [head]
#       plt.clf()
#       plt.imshow(matrix,interpolation='nearest', cmap=cm.inferno)
#       # ax.colorbar() # Add a scale bar
#       # plt.title(p + ' layer ' + str(layer) + ' head ' + str(head))
#       plt.savefig( p + 'layer' + str(layer) + 'head' + str(head)+'.png' )


# GO:0019239;GO:0016814;GO:0016810;GO:0016787;GO:0003824
