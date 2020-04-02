



#### create folders, make config for training
run_option='NoPpiNoAaTypeLabelBertAveL12'
base_option='NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1'
base_config='cc'
for onto in bp mf cc ; do
  mkdir /local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess1000
  mkdir /local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess1000/$run_option/$onto
  cd /local/datdb/deepgoplus/deepgoplus.bio2vec.net/data-cafa/data/SeqLenLess1000/$run_option/$onto
  ## COMMENT scp from older files over, this is okay, we auto fix all input numbers
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$base_config/fold_1/2embPpiAnnotE256H1L12I512Set0/$base_option/vocab* . ##!! okay to use @mf, we will reassign number of labels
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$base_config/fold_1/2embPpiAnnotE256H1L12I512Set0/$base_option/config.json .
done


