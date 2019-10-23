



library('reshape2')
library('ggplot2')
library('gridExtra')
library('dendextend')
library('RColorBrewer')

# graphics.off()
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

# setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiGeluE768H1L12I768PretrainLabelDrop0.1')
setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiMutGeluE768H1L12I768PreLabDrop0.1')

num_label = 589
prot = 'O54992 P23109 P9WNC3' # 'O54992 P23109 P9WNC3'# 'P23109' # B3PC73 O54992 P23109
prot = strsplit(prot,"\\s+")[[1]]

# for (p in prot) {
#   plot_list = list()
#   counter = 1
#   for (layer in 0:11){
#     for (head in 0:0){
#       fin = read.csv( paste0(p , 'layer' , layer, 'head',head,'.csv'), header=F )
#       fin = as.matrix(fin)
#       total_len = nrow(fin)
#       colnames(fin) = 1:nrow(fin)
#       fin = t(fin) ## easier ... so that we know row add to 1.
#       fin = log ( fin * 1000 )
#       ## remove CLS and SEP ??
#       # heatmap(fin,scale="none",col = coul,Colv=NA,main=paste('Layer',layer,'Head',head))
#       fin = melt(fin)
#       total_median = quantile ( fin[,3], 0.5 )
#       print (total_median)
#       total_max = max(fin[,3])
#       p1 = ggplot(data = fin, aes(x=Var1, y=Var2, fill=value)) + geom_tile() +
#       geom_vline(xintercept=total_len-num_label) +
#       geom_hline(yintercept=total_len-num_label) +
#       ggtitle(paste0(p , 'layer' , layer, 'head',head)) +
#       scale_fill_gradient2(low = "blue", high = "red") + # midpoint = total_median, limit = c(0,total_max)
#       theme(legend.title = element_blank()) #,axis.title.x=element_blank(),axis.title.y=element_blank(),axis.text.x = element_blank(),axis.text.y = element_blank())
#       png (file = paste0(p,'H',head,'L',layer,'.png'),width=10, height=10, units='in', res = 500)
#       print (p1)
#       dev.off()
#       # plot_list[[counter]] = p1
#       counter = counter + 1
#     }
#   }
# }

# count_high 

ave_sideway = function(fin,num_label){
  ## compute some summary statistics on best "aa"
  total_len = nrow(fin)
  lowbound = quantile(fin,.25)
  down = rowSums(fin > lowbound)
  highQ = quantile(down,0.5)
  if (highQ==max(down)){
    highQ = quantile(down,0.5)
  }
  print ('lowbound')
  print (lowbound)
  print (dim(fin))
  print (length((down)))
  print (summary(down))
  return ( list (down, sort ( which (down > highQ) ) ) )

}

# p='P23109'
layer = 0
num_label = 589
# prot = 'O54992 P23109 P9WNC3' # 'B3PC73 O54992 P23109'# 'P23109' # B3PC73 O54992 P23109
# prot = strsplit(prot,"\\s+")[[1]] O54992
prot = 'O54992 P23109 P9WNC3'
prot = strsplit(prot,"\\s+")[[1]]
for (name in c('layerAA2AA','layerAA2GO','layerAA2all')){ # ,'layerAA2all','layerAA2GO'
  for (p in prot){
    pdf(paste0(p,name,'_ItoJ.pdf'),width=8, height=12)
    par(mfrow=c(6,4))
    # par(mfrow=c(4,3))
    for (layer in 1:11) {
      fin = read.csv( paste0(p , 'layer' , layer, 'head',0,'.csv'), header=F )
      fin = as.matrix(fin)
      num_aa = nrow(fin)-num_label-2 ## CLS and SEP
      fin = fin [,-1*c(1,num_aa+2)] ## remove the weights toward CLS and SEP ?
      fin = fin [-1*c(1,num_aa+2),] ## remove the weights by CLS and SEP ?

      # fin = log ( fin * 1000 )

      if (name == 'layerAA2AA'){
        fin = fin [1:num_aa, 1:num_aa] ## get contribution of aa toward all the aa itself

      } else if ( name=='layerAA2GO') {
        fin = fin [1:num_aa, (num_aa+1):ncol(fin) ] ## get contribution AA --> GO

      } else if ( name=='layerAA2all') {
        fin = fin [1:num_aa, ] ## get contribution toward all the signals

      } else {
        print ('nothing')
        break
      }

      ## re scale 
      fin = fin / rowSums(fin)
      z = ave_sideway (fin,num_label)

      print (p)
      hist(z[[2]],breaks=15,main=paste(p,'layer',layer),xlab='position')
      if (p=='O54992'){
        abline(v = c(51,115,182,337), col='red', lty=2 )
      }

      qqplot(z[[2]]/num_aa, runif(num_aa),main='qqplot',xlab='uniform',ylab='observe',pch=16,cex=.75)
      abline(0,1)

    }
    dev.off()
  }
}

##



