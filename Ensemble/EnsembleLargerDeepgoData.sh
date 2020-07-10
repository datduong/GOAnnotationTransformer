

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/Ensemble'
cd $codepath

main_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/'

onto='mf'

model='NoPpiYesAaTypeLabelBertAveL12Epo1000bz6'

# prediction1=$main_path/$model/$onto/prediction_train_all_on_test.pickle
# save_file=$main_path/$model/$onto/'EnsembleMetaGO.E10.'

where_out='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/DeepgoplusSingleModel/'
prediction1=$where_out/predictions.$onto.numpy #! single model of deepgocnn
save_file=$where_out/predictions.$onto'.EnsembleMetaGO.'


prediction2=$main_path/'bonnie+motif'/MetaGO/blastPsiblastResultEval10/test-$onto-prediction.pickle

python3 EnsembleSimple.py $prediction1 $prediction2 $save_file



