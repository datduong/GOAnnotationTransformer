
import pickle,os,sys,re
import pandas as pd
import numpy as np


#! smin score, follow deepgoplus paper.
#! requires IC score...

main_dir = '/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000'
model_path = 'NoPpiYesAaTypeLabelBertAveL12Epo1000bz6'
onto_name = 'mf'
pathout = main_dir+'/'+model_path+'/'+onto_name+'/'+'test_panda_deepgo_format.pickle'

os.chdir(main_dir)

####

def MakeLabel1hot (true_label_set,label_tested_dict,num_label=1):
  array = np.zeros(num_label)
  where = np.array ( [ label_tested_dict[label] for label in true_label_set ] )
  array[where] = 1
  return list(array)


#! need to format our matrix output into a panda data frame

prediction_matrix = pickle.load(open(main_dir+'/'+model_path+'/'+onto_name+'/prediction_train_all_on_test.pickle',"rb"))
true_matrix = prediction_matrix['true_label'] ##!! make sure to check keys. blast might be using different @true_label
prediction_matrix = prediction_matrix['prediction']

label_tested = pd.read_csv(main_dir+'/Label.'+onto_name+'.tsv',header=None,sep='\t') ##!! do not need to reorder, we already sorted by names
label_tested = list (label_tested[0]) ## panda format, take first col
label_tested_dict = {label : index for index, label in enumerate(label_tested)}

num_prot,num_label = prediction_matrix.shape

input_file_text = pd.read_csv(main_dir+'/bonnie+motif/test-'+onto_name+'.tsv',header=None,sep='\t') ## need protein names
protein_name = list ( input_file_text[0] ) 
true_label_set = list (input_file_text[2] )

annotations = []
label_1hot = []
prediction_list = [] ##!! list of array

for index,prot in enumerate(protein_name):
  this_label = re.sub("GO","GO:",true_label_set[index]) ##!! need {GO:abc, GO:xyz}
  this_label = set ( this_label.split() )
  annotations.append(this_label) # append set, later, we make this into a column
  label_1hot.append( true_matrix[index] ) # we already have 1 hot
  prediction_list.append( prediction_matrix[index] )


data = {'proteins':protein_name,
        'annotations':annotations,
        'labels':label_1hot,
        'preds':prediction_list
        #! probably don't need sequence
        }


df = pd.DataFrame(data)

df.to_pickle(pathout)
