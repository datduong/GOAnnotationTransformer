
## using the newer code ... hopefully the fp16 works


## LM tune the data
# conda activate tensorflow_gpuenv
server='/local/datdb'
data_dir=$server/'deepgo/data/DataToFinetuneBertTokenPredict/FinetunePhaseData'
mkdir $server/'deepgo/data/BertNotFtAARawSeqGO'

for ontology in 'cc' ; do 
  last_save=$server/'deepgo/data/BertNotFtAARawSeqGO/fold_1'$ontology
  output_dir=$server/'deepgo/data/BertNotFtAARawSeqGO/fold_1'$ontology
  mkdir $output_dir

  # bert_vocab=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/vocab+3kmer+GO.txt'
  bert_vocab=$output_dir/'vocabAA+GO.txt'
  # config_name=$server/'BERTPretrainedModel/cased_L-12_H-768_A-12Kmer2016/config.json'
  config_name=$output_dir/config.json

  train_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/train-'$ontology'-aa.csv'
  eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/dev-'$ontology'-aa.csv'
  label_2test='/local/datdb/deepgo/data/train/deepgo.'$ontology'.csv'

  cd $server/BertGOAnnotation/finetune/

  # 5040 batches train
  ## continue training 
  CUDA_VISIBLE_DEVICES=2 python3 -u run_token_classify.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 100 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 8 --config_name $config_name --do_train --model_type bert --overwrite_output_dir --save_steps 5000 --logging_steps 5000 --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --config_override --learning_rate 0.0001 --seed 2019 > $output_dir/train_point.txt # --no_cuda


  ## testing phase 
  
  eval_masklm_data='/local/datdb/deepgo/data/train/fold_1/TokenClassify/test-'$ontology'-aa.csv'
  CUDA_VISIBLE_DEVICES=2 python3 -u run_token_classify.py --block_size 1792 --mlm --bert_vocab $bert_vocab --train_data_file $train_masklm_data --output_dir $output_dir --num_train_epochs 50 --per_gpu_eval_batch_size 12 --config_name $config_name --do_eval --model_type bert --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_masklm_data --label_2test $label_2test --config_override --eval_all_checkpoints > $output_dir/eval_all_check_point.txt

done 


