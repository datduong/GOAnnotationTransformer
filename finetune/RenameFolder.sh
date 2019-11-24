
for onto in mf cc bp ; do 
cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiYesTypeEp10e10Drop0.1
mkdir ArchiveTrainWithMissingData/
mv * ArchiveTrainWithMissingData/
scp ArchiveTrainWithMissingData/vocab* ArchiveTrainWithMissingData/config.json .
done 


YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0


for onto in mf cc bp ; do 
  mkdir /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/vocab* .
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/config.json .
done 


