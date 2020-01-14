

#### archive previous unused models
for onto in mf bp cc ; do
cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeEp10e10Drop0.1
mkdir ArchiveTrainWithMissingData/
mv * ArchiveTrainWithMissingData/
scp ArchiveTrainWithMissingData/vocab* ArchiveTrainWithMissingData/config.json .
done


#### create folders, make config for training
for onto in mf cc bp ; do
  mkdir /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/vocab* . ##!! okay to use @mf, we will reassign number of labels
  scp /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge/YesPpi100YesTypeScaleFreezeBert12Ep10e10Drop0.1/config.json .
done

#### scp between servers
for onto in mf cc bp ; do
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/
  scp -r $nlp9:/local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0 .
done

#### scp between servers
where='/local/datdb/deepgo/data/BertNotFtAARawSeqGO'
cd $where
for onto in mf bp cc ; do
  cd /local/datdb/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge
  scp -r YesPpiYes* $hoffman2:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/ProtAnnotTypeLarge
  # scp -r No*Yes* $hoffman2:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
  # scp -r No*No* $hoffman2:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
  # scp -r Yes*100*No* $nlp9:$localdir/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/
done

#### scp between local computer

mkdir /cygdrive/e/BertNotFtAARawSeqGO
for onto in mf cc bp ; do
  mkdir /cygdrive/e/BertNotFtAARawSeqGO/$onto
  mkdir /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1
  mkdir /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
done 

for onto in mf cc bp ; do
  cd /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
  scp -r $hoffman:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/*pdf /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1/
done

for onto in mf cc bp ; do
  cd /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
  scp -r $hoffman:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/SeeAttention/Q9UHD2/*png /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/SeeAttention/Q9UHD2
done


#### scp blast results to local 
mkdir /cygdrive/e/BertNotFtAARawSeqGO/dataExpandGoSet
cd /cygdrive/e/BertNotFtAARawSeqGO/dataExpandGoSet
scp -r $hoffman:$scratch/deepgo/dataExpandGoSet/train/fold_1/blastPsiblastResultEval10* .

mkdir /cygdrive/e/BertNotFtAARawSeqGO/
cd /cygdrive/e/BertNotFtAARawSeqGO/
scp -r $hoffman:$scratch/deepgo/data/train/fold_1/blastPsiblastResultEval10* .


