
# conda activate tensorflow_gpuenv
server='/local/datdb'

## download this file from google drive link provided on github
pretrained_label_path='/local/datdb/deepgo/data/BertMeanLayer12Dim256/label_vector.pickle'

## model name 
## you can use NoPpiYesAaTypePreTrainBertLabel to apply only Motif data
choice='NoPpiYesAaTypeLabelBertAveL12' 

## suppose you chose NoPpiYesAaTypePreTrainBertLabel, then you must turn off "ppi" mode into "noppi"
model_type='noppi' ##!! noppi--> not using ppi, and ppi--> uses extra data

## save data in pickle to avoid pre-processing, this called cache_name in Transformer code
cache_name='DataWithMotifInPickle' 

save_every=7000

## define parameters for mf-ontology
checkpoint=670152 ## use this when we want to test a specific checkpoint
block_size=2750 ## max len of amino+num_label, mf and cc 1792 but bp has more term 2048

batch_size=1
seed=2020

for ontology in cc ; do 

  if [[ $ontology == 'cc' ]]
  then
    seed=1998 # 2020 ##!! we switch seed so that we can train at batch=4, we tried seed=2019 but kept on getting mem error for some unlucky batch
    # batch_size=2
    # block_size=1792
    checkpoint=1105830
  fi

  if [[ $ontology == 'bp' ]]
  then
    seed=2019
    batch_size=2
    block_size=2048 ## have more bp labels
    checkpoint=65475
  fi

  output_dir=$server'/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/'$choice'/'$ontology
  # mkdir $output_dir
  bert_vocab=$output_dir/vocabAA.txt ## see example file in github
  config_name=$output_dir/config.json

  #### download from google drive, and replace paths here.
  data_text_location='/local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000'
  aa_type_file=$data_text_location/bonnie+motif/'train_'$ontology'_prot_annot_type_count.pickle' ## Domain info found in uniprot for train data
  train_data_file=$data_text_location/bonnie+motif/train-$ontology.tsv ## okay to call it as long as it has ppi
  eval_data_file=$data_text_location/bonnie+motif/test-$ontology.tsv
  label_2test=$data_text_location/Label.$ontology.tsv

  cd $server/GOAnnotationTransformer/TrainModel/

  #### train the model

  ## continue training use @model_name_or_path and turn off @config_override
  ## add --pretrained_label_path $pretrained_label_path so that we run with pretrained GO embeddings
  ## add --aa_type_file $aa_type_file --reset_emb_zero when we have domain/motif info

  ## suppose you chose NoPpiYesAaTypePreTrainBertLabel, then you will keep the flag --aa_type_file $aa_type_file --reset_emb_zero
  ## suppose to run Base Transformer without any extra information, then you remove --aa_type_file $aa_type_file --reset_emb_zero
  ## suppose you want to train end-to-end and not used a pre-trained GO embeddings, then you remove --pretrained_label_path $pretrained_label_path

  CUDA_VISIBLE_DEVICES=6 python3 -u RunTokenClassifyProtData.py --aa_block_size 2048 --train_dev_fraction 0.90 --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --num_train_epochs 100 --per_gpu_train_batch_size $batch_size --per_gpu_eval_batch_size 2 --config_name $config_name --do_train --model_type $model_type --overwrite_output_dir --save_steps $save_every --logging_steps $save_every --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --learning_rate 0.0001 --seed $seed --fp16 --config_override --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero > $output_dir/train_point.txt


  # #### testing phase
  # for test_data in 'test'  ; do # 'dev'

  #   ##!! normal testing on same set of labels

  #   ## we use exactly same arguement settings as training, except for --eval_all_checkpoints --checkpoint $checkpoint
  #   ## use --aa_type_file $aa_type_file --reset_emb_zero to use domain/motif
  #   ## use --eval_all_checkpoints to eval all checkpoints
  #   ## --checkpoint $checkpoint will evaluate at only exactly one checkpoint

  #   save_prediction='prediction_train_all_on_'$test_data
  #   # eval_data_file='/local/datdb/deepgo/data/train/fold_1/ProtAnnotTypeData/'$test_data'-'$ontology'-input.tsv'
  #   CUDA_VISIBLE_DEVICES=0 python3 -u RunTokenClassifyProtData.py --aa_block_size 2048 --train_dev_fraction 0.9 --save_prediction $save_prediction --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size $batch_size --config_name $config_name --do_eval --model_type $model_type --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --eval_all_checkpoints --checkpoint $checkpoint --pretrained_label_path $pretrained_label_path > $output_dir/'eval_'$test_data'_check_point.txt'

  #   # ##!! do zeroshot on larger set
  #   # save_prediction='save_prediction_expand'
  #   # eval_data_file='/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/fold_1/ProtAnnotTypeData/'$test_data'-'$ontology'-input.tsv'
  #   # label_2test='/local/datdb/deepgo/dataExpandGoSet16Jan2020/train/deepgo.'$ontology'.csv' ## COMMENT larger label set

  #   # ## define params for mf-ontology
  #   # new_num_labels=1697 ##!! more labels than original set
  #   # block_size=2816 ##!! zeroshot need larger block size because more labels

  #   # if [[ $ontology == 'cc' ]]
  #   # then
  #   #   new_num_labels=989
  #   #   block_size=2816
  #   # fi

  #   # if [[ $ontology == 'bp' ]]
  #   # then
  #   #   new_num_labels=2980
  #   #   block_size=4048
  #   # fi

  #   # model_name_or_path=$output_dir/'checkpoint-'$checkpoint ##!! load in checkpoint, then replace emb for correct size
  #   # CUDA_VISIBLE_DEVICES=0 python3 -u RunTokenClassifyProtData.py --model_name_or_path $model_name_or_path --new_num_labels $new_num_labels --save_prediction $save_prediction --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size $batch_size --config_name $config_name --do_eval --model_type $model_type --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --config_override --eval_all_checkpoints --checkpoint $checkpoint --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero > $output_dir/'eval_'$test_data'_expand.txt'

  # done

  # #### see attention map

  # cd $server/GOAnnotationTransformer/SeeAttention/
  # model_name_or_path=$output_dir/'checkpoint-'$checkpoint ##!!##!!
  # for test_data in 'train' 'test' ; do
  #   eval_data_file='/local/datdb/deepgo/data/train/fold_1/ProtAnnotTypeData/'$test_data'-'$ontology'-input.tsv'

  #   ## we get attention of only some proteins, not for every proteins
  #   name_get_attention=$server/GOAnnotationTransformer/SeeAttention/name_get_attention_$test_data.tsv ## (github has same file name)

  #   CUDA_VISIBLE_DEVICES=0 python3 -u ViewAttention.py --model_name_or_path $model_name_or_path --name_get_attention $name_get_attention --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size $batch_size --config_name $config_name --do_eval --model_type $model_type --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --eval_all_checkpoints --checkpoint $checkpoint --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero > $output_dir/'see_att_check_point.txt'
  # done

  # #### get hidden vec of GOs

  # cd $server/GOAnnotationTransformer/TrainModel/
  # model_name_or_path=$output_dir/'checkpoint-'$checkpoint
  # for test_data in 'test' ; do ## able to do on 'train' ... but let's not worry about train data
  #   eval_data_file='/local/datdb/deepgo/data/train/fold_1/ProtAnnotTypeData/'$test_data'-'$ontology'-input.tsv'
  #   govec_hidden_name=$test_data'_govec_hidden_layer'
  #   CUDA_VISIBLE_DEVICES=0 python3 -u GOVecHiddenLayerMean.py --model_name_or_path $model_name_or_path --govec_hidden_name $govec_hidden_name --cache_name $cache_name --block_size $block_size --mlm --bert_vocab $bert_vocab --train_data_file $train_data_file --output_dir $output_dir --per_gpu_eval_batch_size $batch_size --config_name $config_name --do_eval --model_type $model_type --overwrite_output_dir --evaluate_during_training --eval_data_file $eval_data_file --label_2test $label_2test --eval_all_checkpoints --checkpoint $checkpoint --pretrained_label_path $pretrained_label_path --aa_type_file $aa_type_file --reset_emb_zero > $output_dir/'govec_hidden_check_point.txt'
  # done

done



