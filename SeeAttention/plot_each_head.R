

library('reshape2')
library('ggplot2')
library('gridExtra')
library('dendextend')
library('RColorBrewer')

# graphics.off() 
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/fold_1mf2embGeluE768H4L10I768')

num_label = 589
p = 'O54992' # B3PC73 O54992

plot_list = list() 
counter = 1
for (layer in 0:9){ 
  for (head in 0:3){
    fin = read.csv( paste0(p , 'layer' , layer, 'head',head,'.csv'), header=F )
    total_len = nrow(fin) 
    colnames(fin) = 1:nrow(fin)
    fin = log ( as.matrix(fin) * 1000 )
    ## remove CLS and SEP ?? 
    # heatmap(fin,scale="none",col = coul,Colv=NA,main=paste('Layer',layer,'Head',head))
    fin = melt(fin)
    total_median = quantile ( fin[,3], 0.5 ) 
    print (total_median)
    total_max = max(fin[,3])
    p1 = ggplot(data = fin, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
    geom_vline(xintercept=total_len-num_label) +
    geom_hline(yintercept=total_len-num_label) +
    ggtitle(paste0(p , 'layer' , layer, 'head',head)) +
    scale_fill_gradient2(low = "blue", high = "red") + # midpoint = total_median, limit = c(0,total_max)
    theme(legend.title = element_blank()) #,axis.title.x=element_blank(),axis.title.y=element_blank(),axis.text.x = element_blank(),axis.text.y = element_blank()) 
    png (file = paste0(p,'H',head,'L',layer,'.png'),width=16, height=16, units='in', res = 500)
    print (p1)
    dev.off() 
    # plot_list[[counter]] = p1 
    counter = counter + 1 
  }
}


# png (file = paste0(p,'h0h1.png'),width=30, height=10, units='in', res = 400)
# grid.arrange(grobs = plot_list, ncol=10, nrow=2) ## display plot
# # ggsave(file = paste0(p,'.png'), arrangeGrob(grobs = plot_list, ncol=4, nrow=10), width=30, height=10, units='in')  ## save plot
# dev.off() 

