
for onto in mf bp cc ; do 
cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeEp10e10Drop0.1
mkdir ArchiveTrainWithMissingData/
mv * ArchiveTrainWithMissingData/
scp ArchiveTrainWithMissingData/vocab* ArchiveTrainWithMissingData/config.json .
done 

for onto in bp cc ; do 
cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/
scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/vocab* .
scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/config* .
done 


YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0


for onto in mf cc bp ; do 
  mkdir /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiMuhaoNoTypeScaleFreezeBert12Ep10e10Drop0.1
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiMuhaoNoTypeScaleFreezeBert12Ep10e10Drop0.1
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/vocab* .
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/config.json .
done 


