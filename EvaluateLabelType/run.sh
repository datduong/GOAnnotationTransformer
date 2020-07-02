


#### ! BLAST split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/MetaGO/'

for onto in 'cc' 'mf' 'bp'; do
  for model in 'blastPsiblastResultEval10' 'blastPsiblastResultEval100' ; do # 'BaseExtraLayer'

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'$onto'.tsv'

    output=$model_path/$model
    load_path=$output/test-$onto-prediction.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.Jun30.TheirFmax.RmRoot.txt

  done
done
cd $output



#### ! Transformer. split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

for onto in 'mf' ; do # 'bp'
  for model in 'NoPpiYesAaTypeLabelBertAveL12Epo1000bz6' ; do # 'BaseExtraLayer'

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'$onto'.tsv'

    model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/'
    output=$model_path/$model/$onto
    load_path=$output/prediction_train_all_on_test.pickle  # prediction_train_all_on_test.pickle EnsembleMetaGO.E100.max.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.Jun30.TheirFmax.RmRoot.txt
    # $output/$onto.EnsembleMetaGO.E100.max.txt $output/$onto.printout.Jun30.txt

  done
done
cd $output

# ensemble helps, but R@k is not great (only small improvement)
# best improvement happens at mid range 25-75 for both cc and mf.



#### ! run on numpy output of original deepgoplus, single ontology but trained on all 3

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
onto='mf'
label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'$onto'.tsv'
load_path='/u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.pickle'
python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > /u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.accuracy.txt
vim /u/scratch/d/datduong/deepgoplus/data-cafa/predictions.numpy.accuracy.txt



#### ! run on numpy output of original deepgoplus, single model

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
onto='mf'
where_out='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/DeepgoplusSingleModel/'
label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/DataDelRoot/SeqLenLess2000/Label.'$onto'.tsv'
load_path=$where_out/predictions.mf.numpy
python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $where_out/predictions.mf.txt
vim $where_out/predictions.mf.txt

