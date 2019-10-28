

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1')
# average sknewness and KL 

fin = read.table("2embPpiGeluE768H1L12I768PretrainLabelDrop0.1/HistogramValidate/attention_summary.txt",sep='\t',head=T)
# prot  layer head  KL  skewness  prob_mut

save = matrix(0,nrow=12,ncol=6)
for (layer in 0:11){
  for (head in 0:0){
    fin2 = fin [ fin$layer==layer & fin$head==head, ]
    fin2 = apply(fin2[c('KL','skewness')], 2, quantile, c(.25,.5,.75))
    # print (paste('layer',layer,'head',head))
    # print (fin2)
    save[layer+1,] = fin2
  }
}

rownames(save) = paste('layer',0:11)
print (save)
