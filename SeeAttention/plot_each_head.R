

library('reshape2')
library('ggplot2')
library('gridExtra')
library('dendextend')
library('RColorBrewer')

# graphics.off()
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

for (model in c('NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1','NoPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1')){

  setwd(paste0('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/',model,'/SeeAttention'))

  num_label = 589
  #### plot these proteins
  # P76245 P18431 P34283 P62380 O54992 Q5VV41 P39935 P20933
  prot = 'Q9UHD2'
  prot = strsplit(prot,"\\s+")[[1]]

  for (p in prot) {
    print (p)
    for (head in c(0)){
      plot_list = list()
      for (layer in 0:11){
        fin = read.csv( paste0(p, '/', p , '_layer_' , layer, '_head_',head,'.csv'), header=F )
        fin = as.matrix(fin)
        total_len = nrow(fin)
        colnames(fin) = 1:nrow(fin)
        fin = t(fin) ##!!##!! so that we know row add to 1.
        # fin = log( fin * 1000 )
        ## remove CLS and SEP ??
        ## heatmap(fin,scale="none",col = coul,Colv=NA,main=paste('Layer',layer,'Head',head))
        fin = melt(fin)
        total_median = quantile ( fin[,3], 0.5 )
        print (total_median)
        total_max = max(fin[,3])
        #### should cap the high numbers
        cap_off = quantile ( fin[,3], 0.9 )
        p1 = ggplot(data = fin, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
        geom_vline(xintercept=total_len-num_label) +
        geom_hline(yintercept=total_len-num_label) +
        ggtitle(paste0(p , ' Layer ' , layer+1, ' Head ',head+1)) +
        xlab("Attention toward x") + ylab("Attention from x") +
        scale_fill_gradient2(low = "white", high = "black", midpoint = total_median, limit = c(0,cap_off)) + # midpoint = total_median, limit = c(0,total_max) theme(legend.title = element_blank()) + 
        theme_bw() + guides(fill = FALSE) + theme(legend.position = "none")  #,axis.title.x=element_blank(),axis.title.y=element_blank(),axis.text.x = element_blank(),axis.text.y = element_blank())
        # png (file = paste0(p, '/', p, '_head_',head,'_layer_',layer,'_raw.png'),width=4, height=4, units='in', res = 400)
        # print (p1)
        # dev.off()
        #### add boundary for regions
        if (prot=='Q9UHD2'){
          # COILED 407-657;COILED 658-713;DOMAIN protein kinase 9-310;DOMAIN ubiquitin-like 309-385
          p1 = p1 + geom_vline(xintercept=c(9,310,385,407,713), linetype="dashed", color = "red", size=.5)
          p1 = p1 + geom_hline(yintercept=c(9,310,385,407,713), linetype="dashed", color = "red", size=.5)
        }
        plot_list[[layer+1]] = p1
      }
      # grid.arrange(grobs = plot_list, ncol = 3) ## display plot
      ggsave( file = paste0(p, '/', p, '_head_',head,'_raw.png'), arrangeGrob(grobs = plot_list, ncol = 4), width = 11, height = 8, units = c("in") )  ## save plot
    }
  }
}


