import sys,re,os,pickle
import numpy as np
import pandas as pd

sys.path.append("/local/datdb/BertGOAnnotation")
import KmerModel.TokenClassifier as TokenClassifier

sys.path.append("/local/datdb/BertGOAnnotation/finetune")
import evaluation_metric
import PosthocCorrect


def eval (in_dict,sub_array=None):
  prediction = prediction_dict['prediction']
  true_label = prediction_dict['true_label']
  if sub_array is not None:
    prediction = prediction [ : , sub_array ] ## obs x label
    true_label = true_label [ : , sub_array ]
  #
  result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=[5,10,15,20,25])
  return result

## check accuracy of labels not seen in training.
## pure zeroshot approach.

# onto = 'bp'

for onto in ['cc','mf','bp']:

  print ('\n\ntype {}'.format(onto))

  label_original = pd.read_csv('/local/datdb/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  label_original = set(list(label_original[0]))

  label_large = pd.read_csv('/local/datdb/deepgo/dataExpandGoSet/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  label_large = set(list(label_large[0]))

  label_unseen = sorted ( list ( label_large - label_original ) )
  label_large = sorted(label_large) ## by default we sort label for the model

  label_lookup = {value:counter for counter,value in enumerate(label_large)}
  label_unseen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_unseen ] )
  label_seen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_original ] )

  ## want to compute accuracy on original set of labels, then on unseen labels
  ## possible original set prediction will change because we do joint prediction. so attention weight will affect outcome

  prediction_dict = pickle.load(open("/local/datdb/deepgo/data/BertNotFtAARawSeqGO/"+onto+"/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/prediction_train_all.pickle","rb"))

  print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))

  print ('\nwhole')
  evaluation_metric.print_metrics( eval(prediction_dict) )

  print('\noriginal')
  evaluation_metric.print_metrics( eval(prediction_dict, label_seen_pos) )

  print ('\nunseen')
  evaluation_metric.print_metrics( eval(prediction_dict, label_unseen_pos) )
