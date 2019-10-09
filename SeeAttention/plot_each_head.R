

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
  for (head in 0:1){
    fin = read.csv( paste0(p , 'layer' , layer, 'head',head,'.csv'), header=F ) 
    fin = as.matrix(fin)
    ## remove CLS and SEP ?? 
    # heatmap(fin,scale="none",col = coul,Colv=NA,main=paste('Layer',layer,'Head',head))
    fin = melt(fin)
    p1 = ggplot(data = fin, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
    ggtitle(paste0(p , 'layer' , layer, 'head',head)) +
    scale_colour_gradient(low = "white", high = "black") +
    theme(legend.title = element_blank(),axis.title.x=element_blank(),axis.title.y=element_blank(),axis.text.x = element_blank(),axis.text.y = element_blank(),axis.ticks = element_blank()) 
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

