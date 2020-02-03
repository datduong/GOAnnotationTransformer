
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAAseqGO'

for ontology in 'mf' ; do 
  last_save=$server/'deepgo/data/BertNotFtAAseqGO/fold_1'$ontology
  output_dir=$server/'deepgo/data/BertNotFtAAseqGO/fold_1'$ontology
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocab.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json

  train_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/train-'$ontology'.csv'
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/dev-'$ontology'.csv'
  label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/BertGOAnnotationTrainModel/

  # 5040 batches train
  ## continue training 
  # CUDA_VISIBLE_DEVICES=5 python3 -u run_token_classify.py --block_size 2048 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 100 --per_gpu_train_batch_size 6 --per_gpu_eval_batch_size 10 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 3000 --logging_steps 3000 --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --learning_rate 0.0001 > $output_dir/train_point.txt # --no_cuda


  ## testing phase 
  
  eval_data_file='/local/datdb/deepgo/data/train/fold_1/TokenClassify/test-'$ontology'.csv'
  CUDA_VISIBLE_DEVICES=5 python3 -u run_token_classify.py --block_size 2048 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 50 --per_gpu_eval_batch_size 12 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --eval_all_checkpoints > $output_dir/eval_all_check_point.txt

done 


