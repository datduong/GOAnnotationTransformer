



library('reshape2')
library('ggplot2')
library('gridExtra')
library('dendextend')
library('RColorBrewer')

# graphics.off()
coul <- colorRampPalette(brewer.pal(8, "PiYG"))(25)

# setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiGeluE768H1L12I768PretrainLabelDrop0.1/ManualValidate')

setwd('/u/scratch/d/datduong/deepgo/data/BertNotFtAARawSeqGO/mf/fold_1/2embPpiMutGeluE768H1L12I768PreLabDrop0.1/ManualValidate')


ave_sideway = function(fin,num_label){
  ## compute some summary statistics on best "aa"
  lowbound = quantile(fin,.25)
  down = rowSums(fin > lowbound)
  highQ = quantile(down,0.5)
  if (highQ==max(down)){
    highQ = quantile(down,0.5)
  }
  print ('lowbound'); print (lowbound)
  print (dim(fin)) ; print (length((down)))
  print (summary(down))
  return ( list (down, sort ( which (down > highQ) ) ) )
}


layer = 0
num_label = 589
prot = 'O54992 Q6X632 P0A812 Q96B01 Q5VV41 Q6FJA3 Q9HWK6' # Q6X632 P0A812 Q96B01 Q5VV41 Q6FJA3 Q9HWK6
prot = strsplit(prot,"\\s+")[[1]]

for (name in c('layerAA2all','layerAA2GO')){ # ,'layerAA2all','layerAA2GO' 'layerAA2AA'
  for (p in prot){

    pdf(paste0(p,name,'_i2j.pdf'),width=8, height=12)
    par(mfrow=c(6,4))

    for (layer in 1:11) {
      fin = read.csv( paste0(p, '/', p , 'layer' , layer, 'head',0,'.csv'), header=F )
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

      fin = fin / rowSums(fin) ## re scale
      z = ave_sideway (fin,num_label)

      print (p)
      hist(z[[2]],breaks=10,main=paste(p,'layer',layer),xlab='position')
      if (p=='O54992'){
        abline(v = c(51,115,182,337), col='red', lty=2 )
      }

      qqplot(runif(num_aa), z[[2]]/num_aa, main='QQ plot',xlab='Uniform',ylab='Observe',pch=16,cex=.65)
      abline(0,1)

    }
    dev.off()
  }
}

##



