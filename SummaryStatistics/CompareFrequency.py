

import os,sys,re,pickle
import numpy as np
import pandas as pd 

#### count number of label for each protein

#### frequency per GO labels


for onto in ['mf','cc','bp']:
  LabelCount = {}
  NumLabelPerSample = []
  # dataExpandGoSet
  # file_name = "/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/train-"+onto+"-same-origin.tsv"
  file_name = "/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/train-"+onto+"-16Jan20.tsv"
  fin = open(file_name,"r")
  for index,line in enumerate(fin):
    # if index == 0:
    #   continue
    line = line.strip().split('\t')
    label = line[1].split(";")
    NumLabelPerSample.append ( len(label) )
    for l in label:
      if l in LabelCount:
        LabelCount[l] = LabelCount[l] + 1
      else:
        LabelCount[l] = 1
  fin.close()
  ##
  #### original file
  LabelCountOriginal = {}
  file_name = "/u/scratch/d/datduong/deepgo/data/train/train-"+onto+".tsv" 
  fin = open(file_name,"r")
  for index,line in enumerate(fin):
    # if index == 0:
    #   continue
    line = line.strip().split('\t')
    # line[1] = re.sub(":","",label[1])
    label = line[1].split(";")
    NumLabelPerSample.append ( len(label) )
    for l in label:
      if l in LabelCountOriginal:
        LabelCountOriginal[l] = LabelCountOriginal[l] + 1
      else:
        LabelCountOriginal[l] = 1
  fin.close()
  ## compare: 
  print ('onto {}'.format(onto))
  if onto == 'mf': 
    print (LabelCountOriginal['GO0004930'])
    print (LabelCount['GO0004930'])
  print_see_more_counter = 0 
  for k,v in LabelCountOriginal.items(): 
    if LabelCount[k] > v: 
      print_see_more_counter = print_see_more_counter + 1
  #
  print ('print_see_more_counter')
  print (print_see_more_counter)





  print ('\nonto {}'.format(onto))
  print ('Total num label')
  print (len(LabelCount))
  print ('NumLabelPerSample')
  print ( np.quantile(NumLabelPerSample,q=[0.25,.5,.75,.95,1]) )
  print ('GO label counter')
  counter = [v for k,v in LabelCount.items()]
  print ( np.quantile(counter,q=[0.25,.5,.75,.95,1]) )
  #### frequency of added terms
  label_original = pd.read_csv('/u/scratch/d/datduong/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  temp = list(label_original[0])
  label_original = [re.sub(":","",l) for l in temp]
  label_original = set(label_original)
  label_in_dict = set(list(LabelCount.keys()))
  added_label = list (label_in_dict - label_original)
  # print ('Added, unseen GO label counter')
  # counter = [v for k,v in LabelCount.items() if k in added_label]
  # print ( np.quantile(counter,q=[0.25,.5,.75,.95,1]) )
  print ('Original, seen GO label counter')
  counter = [v for k,v in LabelCount.items() if k in label_original]
  print ( np.quantile(counter,q=[0.25,.5,.75,.95,1]) )
  if onto=='cc': 
    print ('see GO:0031983')
    print (LabelCount['GO0031983'])




onto mf
Total num label
1673
NumLabelPerSample
[ 5.  7. 11. 25. 64.]
GO label counter
[1.0000e+01 1.8000e+01 5.2000e+01 4.6120e+02 1.0321e+04]
Added, unseen GO label counter
[ 8. 12. 18. 29. 43.]
Original, seen GO label counter
[   49.     88.    226.5  1094.9 10321. ]

onto cc
Total num label
979
NumLabelPerSample
[ 5.  9. 14. 25. 75.]
GO label counter
[1.200e+01 2.700e+01 9.550e+01 9.184e+02 2.585e+04]
Added, unseen GO label counter
[ 12. 9. 20. 32. 41.]
Original, seen GO label counter
[   58.5   111.    292.5  2380.4 25850. ]

onto bp
Total num label
2980
NumLabelPerSample
[ 13.  25.  44. 103. 448.]
GO label counter
[   49.      91.     218.5   1410.05 19232.  ]
Added, unseen GO label counter
[ 42.  60.  94. 145. 265.]
Original, seen GO label counter
[  232.75   365.     860.    3707.75 19232.  ]
