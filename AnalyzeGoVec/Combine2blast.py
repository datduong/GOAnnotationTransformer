import sys,re,os,pickle
import numpy as np
import pandas as pd

from copy import deepcopy
## we can do some adhoc method here to combine to blast. Take max ?


sys.path.append("/u/scratch/d/datduong/BertGOAnnotation")
import TransformerModel.TokenClassifier as TokenClassifier

sys.path.append("/u/scratch/d/datduong/BertGOAnnotationTrainModel")
import evaluation_metric
import PosthocCorrect


def eval (prediction_dict,sub_array=None):
  prediction = prediction_dict['prediction']
  true_label = prediction_dict['true_label']
  if sub_array is not None:
    prediction = prediction [ : , sub_array ] ## obs x label
    true_label = true_label [ : , sub_array ]
  #
  result = evaluation_metric.all_metrics ( np.round(prediction) , true_label, yhat_raw=prediction, k=[5,10,15,20,25,30,35,40])
  return result

def get_acc (prediction_dict,label_seen_pos,label_unseen_pos,prefix=""):
  print ('\nmodel type {}'.format(prefix))
  print ('\nsize {}\n'.format(prediction_dict['prediction'].shape))
  print ('\nwhole')
  evaluation_metric.print_metrics( eval(prediction_dict) )
  print('\noriginal')
  evaluation_metric.print_metrics( eval(prediction_dict, label_seen_pos) )
  print ('\nunseen')
  evaluation_metric.print_metrics( eval(prediction_dict, label_unseen_pos) )


## check accuracy of labels not seen in training.
## pure zeroshot approach.

# onto = 'bp'

for onto in ['cc','mf','bp']:

  print ('\n\ntype {}'.format(onto))

  label_original = pd.read_csv('/u/scratch/d/datduong/deepgo/data/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  label_original = set(list(label_original[0]))

  label_large = pd.read_csv('/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/deepgo.'+onto+'.csv',sep="\t",header=None)
  label_large = set(list(label_large[0]))

  label_unseen = sorted ( list ( label_large - label_original ) )
  label_large = sorted(label_large) ## by default we sort label for the model

  label_lookup = {value:counter for counter,value in enumerate(label_large)}
  label_unseen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_unseen ] )
  label_seen_pos = np.array ( [label_lookup[v] for v in label_lookup if v in label_original ] )

  ## want to compute accuracy on original set of labels, then on unseen labels
  ## possible original set prediction will change because we do joint prediction. so attention weight will affect outcome

  # prediction_dict = pickle.load(open("/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/"+onto+"/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/prediction_train_all.pickle","rb"))

  prediction_dict = pickle.load(open("/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/"+onto+"/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/save_prediction_expand.pickle","rb"))

  get_acc (prediction_dict,label_seen_pos,label_unseen_pos,prefix="nn model")

  prediction_dict_blast = pickle.load(open("/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1/blastPsiblastResultEval10/test-"+onto+"-prediction.pickle","rb"))
  ## mismatch row. must fix, because the new uniprot don't have some of these names anymore
  if onto == 'cc':
    prediction_dict_blast['prediction'] = np.delete(prediction_dict_blast['prediction'], [4265, 4598, 2609, 3586, 7884, 3768], axis=0)
    prediction_dict_blast['true_label'] = np.delete(prediction_dict_blast['true_label'], [4265, 4598, 2609, 3586, 7884, 3768], axis=0)

  if onto == 'mf':
    prediction_dict_blast['prediction'] = np.delete(prediction_dict_blast['prediction'], [4167, 4957, 84], axis=0)
    prediction_dict_blast['true_label'] = np.delete(prediction_dict_blast['true_label'], [4167, 4957, 84], axis=0)

  if onto == 'bp':
    prediction_dict_blast['prediction'] = np.delete(prediction_dict_blast['prediction'], [1182, 7114, 5558, 8289, 2083, 7767], axis=0)
    prediction_dict_blast['true_label'] = np.delete(prediction_dict_blast['true_label'], [1182, 7114, 5558, 8289, 2083, 7767], axis=0)

  get_acc (prediction_dict_blast,label_seen_pos,label_unseen_pos,prefix="blast-psiblast")

  print (prediction_dict['prediction'])
  print (prediction_dict_blast['prediction'])
  ## make max
  prediction_dict['prediction'] = np.maximum( prediction_dict['prediction'] , prediction_dict_blast['prediction'] )
  get_acc (prediction_dict,label_seen_pos,label_unseen_pos,prefix="max ( nn model, blast )")

