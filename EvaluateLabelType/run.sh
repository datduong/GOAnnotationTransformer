


####
### ! Do this on the larger deepgo dataset. Transformer. train on our own large deepgo dataset, and then we see what the result are for the added terms

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

for onto in 'cc' ; do # 'bp'

  for model in 'YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1' ; do # 'BaseExtraLayer' Epo1000bz6

    where='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'$onto'/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge16Jan20/'

    output=$where/$model
    load_path=$output/prediction_train_all_on_test.pickle

    python3 $codepath/EvalLabelUnseen.py $onto $where $model $load_path > $output/$onto.printout.txt

  done
done
cd $output



####
###! we run deepgoplus on the original deepgo data expanded with rare labels

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
for onto in bp ; do # 'cc' 'mf'
  label_path='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/deepgo.'$onto'.csv'
  load_path='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/deepgoplusModel'
  model='predictions_expanddata.'$onto'.numpy'
  model_path=$load_path/$model
  python3 $codepath/EvalLabelUnseen.py $onto $load_path $model $model_path > $load_path/$onto.numpy.accuracy.txt
  # vim $load_path/$onto.numpy.accuracy.txt
done




#### ! BLAST split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/bonnie+motif/MetaGO/'

for onto in 'cc' 'mf' ; do
  for model in 'blastPsiblastResultEval10' 'blastPsiblastResultEval100' ; do # 'BaseExtraLayer'

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'$onto'.tsv'

    output=$model_path/$model
    load_path=$output/test-$onto-prediction.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.txt

  done
done
cd $output



#### ! Transformer. split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

for onto in 'mf' ; do # 'bp'
  for model in 'Yes3dYesAaTypeLabelBertAveL12Frac70' ; do # 'BaseExtraLayer' Epo1000bz6

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'$onto'.tsv'

    model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000'  #'DataDelRoot/SeqLenLess2000/'

    output=$model_path/$model/$onto
    load_path=$output/prediction_train_all_on_test125307.pickle  # prediction_train_all_on_test.pickle EnsembleMetaGO.E100.max.pickle EnsembleMetaGO.E10.max.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.txt
    # $output/$onto.printout.TheirFmax.RmRoot.txt $output/$onto.printout.Jun30.txt

  done
done
cd $output

# ensemble helps, but R@k is not great (only small improvement)
# best improvement happens at mid range 25-75 for both cc and mf.


#### ! run on numpy output of original deepgoplus, test single ontology but was trained on all 3

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
onto='mf'
label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'$onto'.tsv'
load_path='/u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.pickle'
python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > /u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.accuracy.txt
vim /u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.accuracy.txt



#### ! run on numpy output, train/test single model

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
onto='mf'
where_out='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/DeepgoplusSingleModel/'
label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'$onto'.tsv'
load_path=$where_out/predictions.mf.numpy
# load_path=$where_out/predictions.mf.EnsembleMetaGO.max.pickle #! ensemble max
python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $where_out/predictions.mf.txt # EnsembleMetaGO
cd $where_out

