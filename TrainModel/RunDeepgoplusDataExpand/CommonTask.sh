



#### create folders, make config for training
run_option='NoPpiNoAaTypeLabelBertAveL12'
new_dir='/local/datdb/deepgoplus/ExpandGoSet/cafa3-data/SeqLenLess1000'
base_option='NoPpiNoAaTypeLabelBertAveL12'
base_config='cc'
for onto in mf cc ; do
  mkdir $new_dir
  mkdir $new_dir/$run_option
  mkdir $new_dir/$run_option/$onto
  # cd $new_dir/$run_option/$onto
  ## COMMENT scp from older files over, this is okay, we auto fix all input numbers
  # scp /local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/$base_option/$base_config/vocab* . ##!! okay to use @mf, we will reassign number of labels
  # scp /local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess2000/$base_option/$base_config/config.json .
done


