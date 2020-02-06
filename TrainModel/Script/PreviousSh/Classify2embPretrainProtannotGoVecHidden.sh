
## using the newer code ... hopefully the fp16 works

## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
# data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

# pretrained_label_path='/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear768.Layer12/label_vector.pickle'
pretrained_label_path='/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear256.Layer12/label_vector.pickle'

choice='2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1' # Lr5e-5 Dr0.2
model_type='ppi'
cache_name='YesPpiYesType'

# onto_type_dict = {'mf': 90594, 'bp': 116144, 'cc': 127647}
# onto_type_dict = {'mf': 80528, 'bp': 145180, 'cc': 141830} ## bert12-->freeze-->layernorm

checkpoint=80528 ## 110726
new_num_labels=1697
block_size=2816 #1792 # mf and cc 1792 but bp has more term  2048

save_every=7000 # 9500 10000

govec_outname=GOvecFromModelHiddenLayer12Expand

for ontology in 'mf' 'bp' 'cc' ; do

  if [[ $ontology == 'bp' ]]
  then
    new_num_labels=2980
    block_size=4048 #2048
    checkpoint=145180
  fi

  if [[ $ontology == 'cc' ]]
  then
    new_num_labels=989
    block_size=2816 #1792
    checkpoint=141830
  fi

  last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  mkdir $output_dir

  bert_vocab=$output_dir/'vocabAA.txt'
  config_name=$output_dir/config.json

  aa_type_file='/local/datdb/deepgo/data/train/fold_1/train_'$ontology'_prot_annot_count.pickle'

  train_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/train-'$ontology'-prot-annot.tsv' ## okay to call it as long as it has ppi
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/dev-'$ontology'-prot-annot.tsv'
  # label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'
  label_2test='/local/datdb/deepgo/dataExpandGoSet/train/deepgo.'$ontology'.csv'

  cd $server/GOAnnotationTransformer/TrainModel/

  ## testing phase --pretrained_label_path $pretrained_label_path
  model_name_or_path=$output_dir'/checkpoint-'$checkpoint

  for test_data in 'test' ; do # 'dev'
    # eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/'$test_data'-'$ontology'-prot-annot.tsv'
    eval_data_file='/local/datdb/deepgo/dataExpandGoSet/train/fold_1/ProtAnnotTypeData/'$test_data'-'$ontology'-prot-annot-input.tsv'

    CUDA_VISIBLE_DEVICES=1 python3 -u MeanAttentionGoVec.py --model_name_or_path $model_name_or_path --new_num_labels $new_num_labels --govec_outname $govec_outname$test_data --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size 2 --config_name $config_name --do_eval --model_type $model_type --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --eval_all_checkpoints --fp16 --checkpoint $checkpoint --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero > $output_dir/'make_govec_'$test_data'.txt'
  done  # --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero


done



