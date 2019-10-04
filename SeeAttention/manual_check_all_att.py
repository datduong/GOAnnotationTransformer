import pickle,os,sys,re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/fold_1mf_relu/')
GO2all = pickle.load(open("GO2GO_attention_O54992_B3PC73.pickle","rb"))

go_names = pd.read_csv('/u/scratch/d/datduong/deepgo/data/deepgo.mf.csv',header=None)
go_names = list(go_names[0])

go = '0016410'
go_index = go_names.index('GO:'+go) # 0002039

# xc = {} 
# z = 'GO:0019901;GO:0019900;GO:0019899;GO:0002039;GO:0005515;GO:0005488;GO:0004674;GO:0004672;GO:0016773;GO:0016301;GO:0016772;GO:0016740;GO:0003824'.split(';')
# z = list (set(z))
# xc['O54992'] = [go_names.index(i) for i in z ]
# z = 'GO:0004553;GO:0016798;GO:0016787;GO:0003824'.split(';')
# z = list (set(z))
# xc['B3PC73'] = [go_names.index(i) for i in z ]

for p in ["O54992","B3PC73"]: 
  print ('prot {}'.format(p))
  this_obs = GO2all[p]
  for head in range(6) : # range (6) :
    x = this_obs [ head ][go_index]
    largest_signal_pos = x.argsort()[::-1][:10] ## take the 10 largest signal
    print ('head {}'.format(head))
    print ( [ go_names[signal_pos] for signal_pos in largest_signal_pos ] ) 
    print ( [ x[signal_pos] for signal_pos in largest_signal_pos ] ) 
    # print ( np.quantile ( np.round(x,6), q=[.25,.5,.75,.95] ) )
    # plt.clf()
    # plt.plot(np.arange(len(x)), x , '-')
    # for xc_line in xc[p]: 
    #   plt.axvline(x=xc_line,color='r')
    # plt.savefig( 'go2go'+p+"_"+go+"_"+str(head)+'.png' )
    


fout = open('check_x2.txt','w')
fout.write(",".join(str(z) for z in x)+"\n")
fout.close() 


O54992  GO:0019901;GO:0019900;GO:0019899;GO:0002039;GO:0005515;GO:0005488;GO:0004674;GO:0004672;GO:0016773;GO:0016301;GO:0016772;GO:0016740;GO:0003824
B3PC73  GO:0004553;GO:0016798;GO:0016787;GO:0003824

0016410
0004672

##

m = read.table('check_x2.txt',header=F)
k = strsplit(as.character(m[1,1]),",")
k = sapply(k,as.numeric)
windows()
plot(1:length(k), k,type='l')


'GO:0005506'
>>> x[100]
0.09436785

