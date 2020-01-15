
#### combine something into blast
server='/u/scratch/d/datduong/'
main_dir=$server'deepgo/data/BertNotFtAARawSeqGO'

test_data='test'

# method1=YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1
# YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1 YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1 NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1 


blast=blastPsiblastResultEval10
for method1 in DeepGOFlatSeqProtBase DeepGOFlatSeqOnlyBase ; do 

  for onto in mf cc bp ; do 

    #### transformer predictors
    test_file1=$server/'deepgo/data/train/fold_1/ProtAnnotTypeData/'$test_data'-'$onto'-input-bonnie.tsv'
    prediction1=$main_dir/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/prediction_train_all_on_test.pickle
    header1='none'

    save_file=$main_dir/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/Ensemble$blast
    mkdir $save_file
    save_file=$save_file/$onto'_prediction.pickle'

    #### blast
    test_file2=$server/deepgo/data/train/fold_1/$test_data'-'$onto.tsv
    prediction2=$server/deepgo/data/train/fold_1/$blast/$test_data-$onto-prediction.pickle
    header2='true'

    code_dir=$server/'BertGOAnnotation/Ensemble'
    cd $code_dir

    #### merge
    python3 Ensemble.py $test_file1 $prediction1 $test_file2 $prediction2 $header1 $header2 $save_file > output.txt

    #### tally accuracy scores for the merge prediction
    code_dir=$server/'BertGOAnnotation/AnalyzeGoVec'
    cd $code_dir
    path_out=$main_dir/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/Ensemble$blast
    # onto,count_file,method,path
    label_to_test=$server/'deepgo/data/train/deepgo.'$onto'.csv'
    python3 AnalyzeGoCountAccuracyAny.py $onto $label_to_test $server/deepgo/data/train/fold_1/ $save_file $path_out > $path_out/$onto'_count.txt'
  done 

  ## COMMENT default merge all 3 onto into one file so we can paste into excel
  path_out_mf=$main_dir/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/Ensemble$blast/mf_count.txt
  path_out_cc=$main_dir/cc/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/Ensemble$blast/cc_count.txt
  path_out_bp=$main_dir/bp/fold_1/2embPpiAnnotE256H1L12I512Set0/$method1/Ensemble$blast/bp_count.txt

  #### parse output 
  final_dir=$server/deepgo/data/BertNotFtAARawSeqGO/EvalLabelByGroup/SeenTestSetEnsemble$blast
  mkdir $final_dir
  cd $final_dir
  cat  $path_out_cc $path_out_mf $path_out_bp > $method1'_count.txt'
  python3 $code_dir/ParseOutput.py $method1'_count.txt' > $method1'_count_parse.txt'

done 

