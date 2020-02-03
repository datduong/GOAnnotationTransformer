

library('reshape2')
library('ggplot2')
library('gridExtra')
library('dendextend')
library('RColorBrewer')

# graphics.off()
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE768H1L12I768PreLab/ManualValidate')

num_label = 589
for (p in c('Q6FJA3', 'O54992', 'P0A812', 'Q6X632', 'Q5VV41', 'O35730', 'Q9S9K9', 'Q96B01', 'Q9HWK6') ) {

  plot_list = list()
  counter = 1
  for (layer in 0:9){
    ave_fin = NULL
    for (head in 0:3){
      fin = read.csv( paste0(p , 'layer' , layer, 'head',head,'.csv'), header=F )
      fin = as.matrix(fin)
      total_len = nrow(fin)
      colnames(fin) = 1:nrow(fin)
      fin = t(fin) ## easier ... so that we know row add to 1.
      if (is.null(ave_fin)){
        ave_fin = fin
      } else {
        ave_fin = fin + ave_fin
      }
    }

    ave_fin = log ( ave_fin/4 * 1000 ) ## average over head for one layer
    ## remove CLS and SEP ??
    # heatmap(fin,scale="none",col = coul,Colv=NA,main=paste('Layer',layer,'Head',head))
    fin = ave_fin ## just override to avoid rename
    fin = melt(fin)
    total_median = quantile ( fin[,3], 0.5 )
    print (total_median)
    total_max = max(fin[,3])
    p1 = ggplot(data = fin, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
    geom_vline(xintercept=total_len-num_label) +
    geom_hline(yintercept=total_len-num_label) +
    ggtitle(paste0(p , 'layer' , layer)) +
    scale_fill_gradient2(low = "blue", high = "red") + # midpoint = total_median, limit = c(0,total_max)
    theme(legend.title = element_blank()) #,axis.title.x=element_blank(),axis.title.y=element_blank(),axis.text.x = element_blank(),axis.text.y = element_blank())
    png (file = paste0(p,'AveHead','L',layer,'.png'),width=10, height=10, units='in', res = 500)
    print (p1)
    dev.off()
    # plot_list[[counter]] = p1
    counter = counter + 1

  }

}

