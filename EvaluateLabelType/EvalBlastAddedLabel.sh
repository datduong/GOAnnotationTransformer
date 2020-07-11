


###!
###! we run deepgoplus on the original deepgo data expanded with rare labels

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath
for onto in bp ; do # 'cc' 'mf'
  load_path='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/deepgoplusModel'
  model='predictions_expanddata.'$onto'.numpy'
  model_path=$load_path/$model
  python3 $codepath/EvalLabelUnseen.py $onto $load_path $model $model_path > $load_path/$onto.numpy.accuracy.Less75.txt
  # vim $load_path/$onto.numpy.accuracy.txt
done


#### ! BLAST split by quantile counts, then get accuracy

codepath='/u/scratch/d/datduong/GOAnnotationTransformer/EvaluateLabelType'
cd $codepath

model_path='/u/scratch/d/datduong/deepgo/dataExpandGoSet16Jan2020/train/fold_1/'

for onto in 'bp' ; do
  for model in 'blastPsiblastResultEval10' 'blastPsiblastResultEval100' ; do # 'BaseExtraLayer'

    output=$model_path/$model
    load_path=$output/test-$onto-prediction.pickle

    python3 $codepath/EvalLabelUnseen.py $onto $output $model $load_path > $output/$onto.printout.txt

  done
done
cd $output

