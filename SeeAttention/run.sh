
. /u/local/Modules/default/init/modules.sh
module load R

cd /u/scratch/d/datduong/GOAnnotationTransformer/SeeAttention
Rscript plot_each_head.R 





for onto in mf cc bp ; do
  scp -r $hoffman:$scratch/deepgo/data/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0/No*Yes* /cygdrive/e/BertNotFtAARawSeqGO/$onto/fold_1/2embPpiAnnotE256H1L12I512Set0
done 

