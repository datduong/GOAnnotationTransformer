

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1')
# average sknewness and KL 

fin = read.table("2embPpiAnnotE768H1L12I768PreLab/HistogramValidate/attention_summary_train.txt",sep='\t',head=T)
# prot  layer head  KL  skewness  prob_mut

save = matrix(0,nrow=12,ncol=6)
for (layer in 0:11){
  for (head in 0:0){
    fin2 = fin [ fin$layer==layer & fin$head==head, ]
    fin2 = fin2 [ fin2$KL > 0 , ]
    fin2 = apply(fin2[c('KL','skewness')], 2, quantile, c(.25,.5,.75))
    # print (paste('layer',layer,'head',head))
    # print (fin2)
    save[layer+1,] = fin2
  }
}

rownames(save) = paste('layer',0:11)
print (save)


2embPpiAnnotE768H1L12I768PreLab KL and skewness
               [,1]       [,2]      [,3]        [,4]        [,5]         [,6]
layer 0  0.02013551 0.05101068 0.1166983 -0.20790528 -0.10041627  0.007598178
layer 1  0.02046190 0.05664547 0.1230692 -0.15606454 -0.01563991  0.121105362
layer 2  0.02432499 0.06060716 0.1366128 -0.23218509 -0.08958056  0.061084219
layer 3  0.03063408 0.06798997 0.1363917 -0.27764665 -0.15332502 -0.006537079
layer 4  0.03503506 0.07735527 0.1509978 -0.37704461 -0.21104073 -0.023751594
layer 5  0.02739271 0.06945925 0.1410362 -0.26019846 -0.12675830  0.009901057
layer 6  0.02949599 0.06352235 0.1399116 -0.31847480 -0.17287682 -0.029517786
layer 7  0.05083266 0.09783693 0.1686412 -0.10933410  0.09433618  0.299173590
layer 8  0.03822010 0.08308900 0.1579454 -0.03591565  0.11393831  0.262846728
layer 9  0.03178480 0.07582665 0.1495422 -0.21797958 -0.05516188  0.130287856
layer 10 0.03912955 0.07917430 0.1529074 -0.12462871  0.03607352  0.184936726
layer 11 0.02290543 0.05514018 0.1223316 -0.18900682 -0.06085822  0.066660023
