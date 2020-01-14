

import sys,re,os,pickle
import numpy as np
import pandas as pd

def ReorderRow (test_file,prediction,need_header=None) :
  # @test_file is some raw tsv
  # @prediction is dict [truth,prediction]
  #### we need to re order the row of prediction so that all saved output match

  prediction = pickle.load(open(prediction,"rb"))
  true_label = None
  try:
    true_label = prediction['truth']
  except:
    true_label = prediction['true_label']
  if true_label is None:
    print ('check input prediction keys.')
    exit()

  prediction = prediction['prediction']

  if need_header is None:
    test_file = pd.read_csv(test_file,header=need_header,sep="\t")
    name_original = list ( test_file[0] )
    name_sort = sorted ( list ( test_file[0] ) ) ##!!##!! sort by name so that every input will be consistent
  else:
    test_file = pd.read_csv(test_file,sep="\t")
    name_original = list ( test_file['Entry'] )
    name_sort = sorted ( list ( test_file['Entry'] ) )

  print ('\ninput file\n')
  print (test_file)

  print ('\nname sort')
  print (name_sort[0:10])

  # map_index_sort = { value:index for index,value in enumerate(name_sort) }

  map_index_original = { value:index for index,value in enumerate(name_original) }

  row_order = np.array ([map_index_original[k] for k in name_sort]) # go over sorted name, get the index where it is in the unsorted vector

  print ('\nsee order')
  print (np.array(name_original)[row_order[0:10]])

  prediction = prediction[row_order]
  true_label = true_label[row_order]
  return prediction, true_label

def CheckSameOrder (matrix1,matrix2) :
  for i in range(0,100,4):
    if (np.sum(matrix1[i] - matrix2[i]) != 0) : ## row must be the same for true label, then we know @row_order is correct.
      print ('fail check')
      print (np.sum(matrix1[i] - matrix2[i]))
      break
  return 1

def CombineResult (matrix1,matrix2,option='max') :
  if option == 'max':
    return np.maximum(matrix1,matrix2)
  if option == 'mean':
    return (matrix1+matrix2)/2


def Ensemble (test_file1,prediction1,test_file2,prediction2,header1=None,header2=None,save_file=None):
  prediction1, true_label1 = ReorderRow(test_file1,prediction1,header1)
  prediction2, true_label2 = ReorderRow(test_file2,prediction2,header2)
  if ( CheckSameOrder(true_label1,true_label2) != 1 ):
    print ('fail')
    return 0
  final_prediction = CombineResult(prediction1,prediction2)
  print ('save file {}'.format(save_file))
  pickle.dump ( {'true_label':true_label1, 'prediction':final_prediction} , open ( save_file, 'wb' ) )
  return 1



