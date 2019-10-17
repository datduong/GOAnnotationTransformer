

## find smallest loss

import os, sys, re, pickle
import numpy as np

this_path = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1'
os.chdir(this_path)
# file_list = ['AsIsPpiE768I768H6L8Drop0.2','AsIsE768I768H6L8Drop0.1','AsIsE768I768H6L8Drop0.2']
file_list = os.listdir(this_path)

for f in sorted(file_list):
  best = np.inf
  best_point = 'none'
  try:
    fin = open(f+'/eval_dev_check_point.txt',"r")
  except:
    continue
  for line in fin :
    if '***** Eval results' in line:
      check_point = line.split()[-2]
    if 'eval_loss' in line:
      this_loss = float ( line.split()[-1] )
      if this_loss < best:
        best = this_loss
        best_point = check_point
  fin.close()
  print (f)
  print ('point {} value {} '.format(best_point,best))



AsIsPpiE768I768H6L8Drop0.2
point 70000 value 0.03968845476352033
AsIsE768I768H6L8Drop0.1
point 25000 value 0.04812020396311132
AsIsE768I768H6L8Drop0.2
point 40000 value 0.048758975277462646
2embGeluE768H6L8I768PretrainLabelDrop0.2
point 170000 value 0.04732364633061465
2embPpiAAtokGeluE768H6L8I768PreLabDrop0.2
point 161264 value 0.0370438506268704
2embPpiGeluE768H6L8I768PretrainLabelDrop0.2
point 130000 value 0.037084622805686814
2embPpiGeluE768H6L8I768PretrainLabelDrop0.2Lr5e-5
point 80000 value 0.03960932617349225


2embGeluE768H6L8I768PretrainLabelDrop0.2
point 35000 value 0.057470154203684044
2embPpiGeluE768H6L8I768PretrainLabelDrop0.2
point 55000 value 0.04706858353408999
AsIsE768I768H6L8Drop0.1
point 70000 value 0.05840684705035692
AsIsE768I768H6L8Drop0.2
point 70000 value 0.05887614430899494
AsIsPpiE768I768H6L8Drop0.2
point 60000 value 0.04832301775078347
2embGeluE768H6L8I768PretrainLabelDrop0.1
point 35000 value 0.05710778718120738
2embGeluE768H6L8I768PretrainLabelDrop0.2
point none value inf
2embPpiAAtokGeluE768H6L8I768PreLabDrop0.2
point results value 0.05503926417527068
