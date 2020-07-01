
import pickle,os,sys,re
import pandas as pd
import numpy as np
from copy import deepcopy

import networkx
import obonet


#! smin score, follow deepgoplus paper.
#! requires IC score...

main_dir = '/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000'
model_path = 'NoPpiYesAaTypeLabelBertAveL12Epo1000bz6'
onto_name = 'mf'
pathout = main_dir+'/'+model_path+'/'+onto_name+'/'+'test_panda_deepgo_format.pickle'

os.chdir(main_dir)

####

# def MakeLabel1hot (true_label_set,label_tested_dict,num_label=1):
#   array = np.zeros(num_label)
#   where = np.array ( [ label_tested_dict[label] for label in true_label_set ] )
#   array[where] = 1
#   return list(array)


####

#! read in their prediction, because we need gold annotation
df_their = pd.read_pickle('/u/scratch/d/datduong/deepgoplus/data-cafa/predictions.pkl') #! this is 3,000 something proteins, not 1,000 something


#! ordering of @terms matter. but the question is why?

onto = 'molecular_function'
graph = obonet.read_obo('/u/scratch/d/datduong/deepgoplus/data-cafa/go.obo') # https://github.com/dhimmel/obonet
terms = list( pd.read_pickle('/u/scratch/d/datduong/deepgoplus/data-cafa/terms.pkl')['terms'] ) #! ordering matters for @terms
terms = [t for t in terms if graph.node[t]['namespace'] == onto] #? filter by ontology
print ('number of term in term pickle after filter {}'.format(len(terms)))
terms_position_map = {name:location for location,name in enumerate(terms)}


#! need to format our matrix output into a panda data frame

prediction_matrix = pickle.load(open(main_dir+'/'+model_path+'/'+onto_name+'/prediction_train_all_on_test.pickle',"rb"))
true_matrix = prediction_matrix['true_label'] ##!! make sure to check keys. blast might be using different @true_label
prediction_matrix = prediction_matrix['prediction']

label_tested = pd.read_csv(main_dir+'/Label.'+onto_name+'.tsv',header=None,sep='\t') ##!! do not need to reorder, we already sorted by names
label_tested = list (label_tested[0]) ## panda format, take first col
# label_tested_dict = {label : index for index, label in enumerate(label_tested)}

##!! reorder @label_tested
order_to_match_deepgo_dict = { terms_position_map[t]:index for index,t in enumerate(label_tested) } ## new-->old

def reorder_array (array,order_to_match_deepgo_dict):
  output = np.array(deepcopy(array))
  for i in range(len(array)):
    output[ i ] = array [ order_to_match_deepgo_dict[i] ] ## in new place, put value of old
  return output

# z = reorder_array (label_tested,order_to_match_deepgo_dict) # ! try it

num_prot,num_label = prediction_matrix.shape

input_file_text = pd.read_csv(main_dir+'/bonnie+motif/test-'+onto_name+'.tsv',header=None,sep='\t') ## need protein names
protein_name = list ( input_file_text[0] )
true_label_set = list (input_file_text[2] )

annotations = []
label_1hot = []
prediction_list = [] ##!! list of array
all_proteins_3_onto = []

for i,row in df_their.iterrows():
  #
  all_proteins_3_onto.append(row['proteins'])
  #
  if row['proteins'] in protein_name: #!! found in our prediction
    # this_label = re.sub("GO","GO:",true_label_set[index]) ##!! need {GO:abc, GO:xyz}
    # this_label = set ( this_label.split() )
    # annotations.append( this_label ) # append set, later, we make this into a column
    index = protein_name.index(row['proteins'])
    # label_1hot.append( reorder_array(true_matrix[index],order_to_match_deepgo_dict) ) # we already have 1 hot, #! reorder
    label_1hot.append( true_matrix[index] ) 
    # prediction_list.append( reorder_array(prediction_matrix[index],order_to_match_deepgo_dict) )
    prediction_list.append( prediction_matrix[index] )
  #
  else: #? we keep the same as before.
    label_1hot.append (np.zeros(num_label)) ## all 0, because they got no label
    prediction_list.append (np.zeros(num_label)) ## all 0, because we never predict on them??

data = {'proteins':all_proteins_3_onto,
        # 'annotations':annotations,
        'labels':label_1hot,
        'preds':prediction_list
        #! probably don't need sequence
        }


df = pd.DataFrame(data)

####

del df_their['labels']
del df_their['preds']

df2 = pd.merge(df,df_their, on='proteins')

df2.to_pickle(pathout)


#! double check strict 0
count_nothing_predicted = 0
for i,row in df2.iterrows():
  if row['preds'].sum() == 0 :
    count_nothing_predicted = count_nothing_predicted + 1

#
print ( 'number of strict 0 {}'.format( count_nothing_predicted ) )

