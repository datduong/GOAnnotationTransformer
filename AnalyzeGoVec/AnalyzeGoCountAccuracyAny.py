import sys,re,os,pickle
import numpy as np
import pandas as pd


sys.path.append("/u/scratch/d/datduong/BertGOAnnotation/finetune")
import evaluation_metric

def eval (prediction_dict,sub_array=None,path="",add_name=""):
  prediction = prediction_dict['prediction']
  try: 
    true_label = prediction_dict['truth']
  except: 
    true_label = prediction_dict['true_label']
  if sub_array is not None:
    print ('len label {}'.format(len(sub_array)))
    prediction = prediction [ : , sub_array ] ## obs x label
    true_label = true_label [ : , sub_array ]
  #
  # threshold_fmax=np.arange(0.0001,1,.005)
  result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=[10,20,30,40,50,60,70],path=path,add_name=add_name)
  return result

def get_label_by_count (count_file) :
  count_file = pd.read_csv(count_file,sep="\t") # GO  count
  quantile = np.quantile( list (count_file['count']) , q=[0.25,.75] )
  low = count_file[count_file['count'] <= quantile[0]]
  low = sorted ( list(low['GO']) )
  high = count_file[count_file['count'] >= quantile[1]]
  high = sorted ( list(high['GO']) )
  middle = count_file[ (count_file['count'] > quantile[0]) & (count_file['count'] < quantile[1]) ]
  middle = sorted ( list(middle['GO']) )
  return low, middle, high


def submitJobs (onto,label_original,count_file,method,path):

  # label_original = pd.read_csv('/u/scratch/d/datduong/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  label_original = pd.read_csv(label_original,sep="\t",header=None)
  label_original = sorted(list(label_original[0])) ## we sort labels in training

  #### compute accuracy by frequency

  low, middle, high = get_label_by_count (count_file+'/CountGoInTrain-'+onto+'.tsv')
  low_index = np.array ( [ index for index, value in enumerate(label_original) if value in low ] )
  middle_index = np.array ( [ index for index, value in enumerate(label_original) if value in middle ] )
  high_index = np.array ( [ index for index, value in enumerate(label_original) if value in high ] )

  prediction_dict = pickle.load(open(method,"rb"))

  print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))

  print ('\nwhole {}'.format(onto))
  evaluation_metric.print_metrics( eval(prediction_dict, path=path, add_name='whole'))

  print('\nlow {}'.format(onto))
  evaluation_metric.print_metrics( eval(prediction_dict, low_index, path=path, add_name='low') )

  print ('\nmiddle {}'.format(onto))
  evaluation_metric.print_metrics( eval(prediction_dict, middle_index, path=path, add_name='middle') )

  print ('\nhigh {}'.format(onto))
  evaluation_metric.print_metrics( eval(prediction_dict, high_index, path=path, add_name='high') )


if len(sys.argv)<1: #### run script
	print("Usage: \n")
	sys.exit(1)
else:
	submitJobs ( sys.argv[1] , sys.argv[2] , sys.argv[3] , sys.argv[4] , sys.argv[5] )



