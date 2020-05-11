


#### ! BLAST split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/MetaGO/'

for onto in 'cc' 'mf' 'bp'; do
  for model in 'blastPsiblastResultEval10' 'blastPsiblastResultEval100' ; do # 'BaseExtraLayer'

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'$onto'.tsv'

    output=$model_path/$model
    load_path=$output/test-$onto-prediction.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.printout.txt

  done
done



#### ! Transformer. split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

for onto in 'cc' ; do # 'bp'
  for model in 'NoPpiNoAaTypeLabelBertAveL12' ; do # 'BaseExtraLayer'

    label_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/Label.'$onto'.tsv'

    model_path='/u/scratch/d/datduong/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/'
    output=$model_path/$model/$onto
    load_path=$output/EnsembleMetaGO.E100.max.pickle  # prediction_train_all_on_test.pickle EnsembleMetaGO.E100.max.pickle

    python3 $codepath/EvalLabelQuantile.py $label_path $onto $load_path > $output/$onto.EnsembleMetaGO.E100.max.txt

  done
done
cd $output

# ensemble helps, but R@k is not great (only small improvement)
# best improvement happens at mid range 25-75 for both cc and mf.
