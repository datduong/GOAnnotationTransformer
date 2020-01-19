
/usr/local/cuda-10.1/bin
export PATH="/usr/local/cuda-10.1/bin:$PATH"

. /u/local/Modules/default/init/modules.sh
module load python/3.7.2

#### load back transformer test model, eval on different groups of GOs ... pure zeroshot
main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
##!! can eval on original dataset or on unseen labels
## COMMENT ZEROSHOT eval here.
load_file_name='save_prediction_expand' # prediction_train_all_on_test save_prediction_expand
for run_type in YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 ; do
  method='/fold_1/2embPpiAnnotE256H1L12I512Set0/'$run_type'/'
  code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
  out_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup'
  mkdir $out_dir
  out_dir=$out_dir/ZeroshotNotEnsemble ## COMMENT ZEROSHOT.
  mkdir $out_dir
  cd $code_dir
  python3 AnalyzeGoTypeAccuracy.py $main_dir $method $load_file_name > $out_dir/$run_type.txt
done
cd $out_dir
##!! parse output
for model in YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1; do 
  python3 $code_dir/ParseOutput.py $model.txt > $model'_parse.txt'
done 


#### load back Transformer model trained on small data, eval on rare labels
main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
##!! can eval on original dataset or on unseen labels
load_file_name='prediction_train_all_on_test' # prediction_train_all_on_test save_prediction_expand
for run_type in NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 ; do
  method='/fold_1/2embPpiAnnotE256H1L12I512Set0/'$run_type'/'
  code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
  out_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup'
  mkdir $out_dir
  out_dir=$out_dir/$load_file_name
  mkdir $out_dir
  cd $code_dir
  ##!! use prediction_train_all_on_test
  python3 AnalyzeGoTypeAccuracy.py $main_dir $method prediction_train_all_on_test > $out_dir/$run_type.txt
done
cd $out_dir
##!! parse output
cd /u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup/prediction_train_all_on_test
for model in NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 ; do 
  python3 $code_dir/ParseOutput.py $model.txt > $model'_parse.txt'
done 



#### load back model, eval based on num of frequency, simple evaluation on total known labels
main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
count_file='/u/scratch/d/datduong/deepgo/data/train/fold_1'
##!!
load_file_name='prediction_train_all_on_test' # prediction_train_all_on_test save_prediction_expand
for run_type in YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 ; do
  method='/fold_1/2embPpiAnnotE256H1L12I512Set0/'$run_type'/'
  code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
  out_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup'
  mkdir $out_dir
  out_dir=$out_dir/$load_file_name
  mkdir $out_dir
  cd $code_dir
  python3 AnalyzeGoCountAccuracy.py $main_dir $count_file $method $load_file_name > $out_dir/$run_type'_count.txt'
  python3 $code_dir/ParseOutput.py $out_dir/$run_type'_count.txt' > $out_dir/$run_type'_count_parse.txt'
done
cd $out_dir
##!! parse output
cd /u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup/prediction_train_all_on_test
for model in NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 ; do 
  python3 $code_dir/ParseOutput.py $model.txt > $model'_parse.txt'
done 



#### load back test file for original deepgo model, eval based on num of frequency

#dataExpandGoSet
main_dir='/u/scratch/d/datduong/deepgo/dataExpandGoSet/train/fold_1' ## also where the count file is 
load_file_name='prediction_train_all_on_test' # prediction_train_all_on_test save_prediction_expand
code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
out_dir='/u/scratch/d/datduong/deepgo/dataExpandGoSet/EvalLabelByGroup'
mkdir $out_dir
model_name='DeepGOFlatSeqOnlyBase' # DeepGOFlatSeqProtBase
model_train_name='ExactAsIs'
out_dir=$out_dir/$model_name
mkdir $out_dir
label_to_test=$main_dir'/deepgo.'$onto'.csv'
for onto in mf bp cc ; do
  method=$main_dir/$model_name/$model_train_name/$onto'b32lr0.001RMSprop/prediction_testset.pickle'
  path_out=$out_dir/$onto
  mkdir $path_out
  cd $code_dir
  # onto,label_original,count_file,method,path
  # python3 AnalyzeGoCountAccuracyAny.py $onto $label_to_test $main_dir $method $path_out > $out_dir/$onto'_count.txt'
  ##!! compute on seen vs unseen
  # onto,prediction_dict,save_file_type,path
  python3 AnalyzeGoTypeAccuracyAny.py $onto $method save_prediction_expand $path_out > $out_dir/$onto'_count.txt'
done
cd $out_dir
cat cc_count.txt mf_count.txt bp_count.txt > output_count.txt
python3 $code_dir/ParseOutput.py output_count.txt > output_count_parse.txt





#### COMMENT test on some other deepgo model
main_dir='/local/datdb/deepgo/data/train/fold_1' ## also where the count file is 
load_file_name='prediction_train_all_on_test' # prediction_train_all_on_test save_prediction_expand
code_dir='/local/datdb/BertGOAnnotation/AnalyzeGoVec'
out_dir='/local/datdb/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup'
mkdir $out_dir
model_name='DeepGOFlatSeqProtBase' # DeepGOFlatSeqProtBase/ExactAsIs
model_train_name='ExactAsIs'
out_dir=$out_dir/$model_name
mkdir $out_dir
for onto in mf bp cc ; do
  method=$main_dir/$model_name/$model_train_name/$onto'b32lr0.001RMSprop/prediction_testset.pickle'
  path_out=$out_dir/$onto
  mkdir $path_out
  cd $code_dir
  # onto,count_file,method,path
  python3 AnalyzeGoCountAccuracyAny.py $onto $main_dir $method $path_out > $out_dir/$onto'_count.txt'
done
cd $out_dir
cat cc_count.txt mf_count.txt bp_count.txt > output_count.txt
python3 $code_dir/ParseOutput.py output_count.txt > output_count_parse.txt


#### use blast to eval added term... not pure zeroshot
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
data_type='dataExpandGoSet16Jan2020' # dataExpandGoSet
load_file_name='save_prediction_expand' # prediction_train_all_on_test save_prediction_expand
for method in blastPsiblastResultEval100 blastPsiblastResultEval10 ; do 
  out_dir='/u/scratch/d/datduong/deepgo/'$data_type'/train/fold_1/'$method
  main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
  code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
  cd $code_dir
  python3 AnalyzeGoTypeAccuracyBlast.py $main_dir $method $load_file_name > $out_dir/$method.txt
  cd $out_dir
done
##!! parse output
code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
for model in blastPsiblastResultEval10 blastPsiblastResultEval100 ; do 
  cd /u/scratch/d/datduong/deepgo/$data_type/train/fold_1/$model
  python3 $code_dir/ParseOutput.py $model.txt > $model'_parse.txt'
done 


#### load back test file, eval based on num of frequency ... BLAST
. /u/local/Modules/default/init/modules.sh
module load python/3.7.2
main_dir='/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/'
count_file='/u/scratch/d/datduong/deepgo/data/train/fold_1'
data_type='data' # dataExpandGoSet
load_file_name='prediction_train_all_on_test' # prediction_train_all_on_test save_prediction_expand
for method in blastPsiblastResultEval100 blastPsiblastResultEval10 ; do 
  out_dir='/u/scratch/d/datduong/deepgo/'$data_type'/train/fold_1/'$method
  code_dir='/u/scratch/d/datduong/BertGOAnnotation/AnalyzeGoVec'
  cd $code_dir
  python3 AnalyzeGoCountAccuracyBlast.py $main_dir $count_file $method $load_file_name > $out_dir/$method'_count.txt'
  cd $out_dir
  python3 $code_dir/ParseOutput.py $out_dir/$method'_count.txt' > output_count_parse.txt
done


