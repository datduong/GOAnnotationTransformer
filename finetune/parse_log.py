

## find smallest loss

import os, sys, re, pickle
import numpy as np

MainPath = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO'
MainSetting='2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1Dcay10e2'

for onto in ['mf','cc','bp']: 
  this_path = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'+onto+'/fold_1/'+MainSetting
  os.chdir(this_path)
  #
  best = np.inf
  best_point = 'none'
  try:
    fin = open('train_point.txt',"r")
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
  print ('\ntype {}'.format(onto))
  print ('point {} value {} '.format(best_point,best))


####
####

## view eval file 
import os, sys, re, pickle
import numpy as np

MainPath = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO'
MainSetting='2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1Dcay10e2'

for onto in ['mf','cc','bp']: 
  print ('\ntype {}'.format(onto))
  this_path = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'+onto+'/fold_1/'+MainSetting
  os.chdir(this_path)
  #
  try:
    fin = open('eval_test_check_point.txt',"r")
  except:
    continue
  for line in fin :
    line = line.strip() 
    if 'auc_macro' in line: 
      print (line)
    if 'auc_micro' in line: 
      print (line)
    if 'eval_loss' in line:
      print (line)
    if 'fmax score' in line:
      print (line)
  fin.close()
 
#

