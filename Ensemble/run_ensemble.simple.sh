

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/Ensemble'
cd $codepath

main_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/'

onto='mf'

model='NoPpiYesAaTypeLabelBertAveL12Epo1000bz6'

prediction1=$main_path/$model/$onto/prediction_train_all_on_test.pickle

prediction2=$main_path/MetaGO/blastPsiblastResultEval100/test-$onto-prediction.pickle

save_file=$main_path/$model/$onto/'EnsembleMetaGO.E100.'

python3 EnsembleSimple.py $prediction1 $prediction2 $save_file



