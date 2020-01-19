

#### find smallest loss

import os, sys, re, pickle
import numpy as np

def delete_worst_checkpoint (this_path, best_point): 
  check_save = [ f for f in os.listdir(this_path) if 'checkpoint-' in f ] 
  for check in check_save: 
    num = float ( re.sub("checkpoint-","",check) ) 
    if num != best_point:
      print ('del {}'.format(check)) 
      os.system ('rm -rf ' + check)


#
MainPath = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO' # ProtAnnotTypeLarge ProtAnnotTypeLarge16Jan20
model_setting = ['YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1']
for m in model_setting: 
  MainSetting='2embPpiAnnotE256H1L12I512Set0/'+m
  print ('\n\nsetting {}\n'.format(m))
  for onto in ['cc']:
    #
    best = np.inf
    best_point = 'none'
    try:
      this_path = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'+onto+'/fold_1/'+MainSetting
      os.chdir(this_path)
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
    if best_point != 'none':
      delete_worst_checkpoint (this_path, float(best_point))



#### view eval file

import os, sys, re, pickle
import numpy as np

MainPath = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO'
MainSetting='2embPpiAnnotE256H1L12I512Set0/NoPpi100NoTypeScaleFreezeBert12Ep10e10Drop0.1'

for onto in ['mf','cc','bp']:
  print ('\ntype {}'.format(onto))
  #
  try:
    this_path = '/local/datdb/deepgo/data/BertNotFtAARawSeqGO/'+onto+'/fold_1/'+MainSetting
    os.chdir(this_path)
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

