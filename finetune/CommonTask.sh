

for onto in mf cc bp ; do
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/
  scp -r $nlp9:/local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0 .
done

