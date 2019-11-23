
for onto in mf cc bp ; do 
cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeTopo/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1
mkdir ArchiveTrainWithMissingData/
mv * ArchiveTrainWithMissingData/
scp ArchiveTrainWithMissingData/vocab* ArchiveTrainWithMissingData/config.json .
done 

