
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
# data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

# pretrained_label_path='/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear768.Layer12/label_vector.pickle'
pretrained_label_path='/local/datdb/deepgo/data/cosine.AveWordClsSep768.Linear256.Layer12/label_vector.pickle'

choice='2embPpiAnnotE256H1L12I768Set0/Ep10e10Drop0.1InitUniform/' # Lr5e-5 Dr0.2
block_size=1792 # mf and cc 1792 but bp has more term  2048
save_every=7000 # 9500 10000

for ontology in 'mf' ; do
  last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/'$ontology/'fold_1'/$choice
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocabAA.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json
  # model_name_or_path=$output_dir/checkpoint-30198

  aa_type_file='/local/datdb/deepgo/data/train/fold_1/train_'$ontology'_prot_annot_count.pickle'

  train_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/train-'$ontology'-prot-annot.tsv' ## okay to call it as long as it has ppi
  eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/dev-'$ontology'-prot-annot.tsv'
  label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/BertGOAnnotation/finetune/

  ## !!   NOTICE. lr=5e-5 and batch=3 didn't work well 

  # # 5040 batches train
  # # continue training use @model_name_or_path and turn off @config_override
  CUDA_VISIBLE_DEVICES=6 python3 -u RunTokenClassifyProtData.py --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 100 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 2 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps $save_every --logging_steps $save_every --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --learning_rate 0.0001 --seed 2019 --fp16 --aa_type_emb --aa_type_file $aa_type_file --reset_emb_zero --config_override --pretrained_label_path $pretrained_label_path > $output_dir/train_point2.txt # --no_cuda

  # # ## testing phase --pretrained_label_path $pretrained_label_path

  # checkpoint=70920
  # for test_data in 'test'; do # 'dev' 
  #   eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/'$test_data'-'$ontology'-prot-annot.tsv'
  #   CUDA_VISIBLE_DEVICES=5 python3 -u RunTokenClassifyProtData.py --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --per_gpu_eval_batch_size 2 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --config_override --eval_all_checkpoints --fp16 --aa_type_emb --aa_type_file $aa_type_file --checkpoint $checkpoint > $output_dir/'eval_'$test_data'_check_point.txt'
  # done  # --pretrained_label_path $pretrained_label_path 

  ## view weights ?? 

  # cd $server/BertGOAnnotation/SeeAttention/
  # data_type='train'
  # eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/TwoEmb/'$data_type'-'$ontology'-prot-annot.tsv'

  # model_name_or_path=$output_dir'/checkpoint-35231'

  # # view_weight_aa_2emb view_weight_aa_skewness
  # CUDA_VISIBLE_DEVICES=6 python3 -u view_weight_aa_2emb.py --data_type $data_type --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --per_gpu_eval_batch_size 2 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --model_name_or_path $model_name_or_path --aa_type_emb --aa_type_file $aa_type_file > $output_dir/view_aa_weights_skew_$data_type.txt # --pretrained_label_path $pretrained_label_path


done


