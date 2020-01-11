library(Rtsne)
library('ggrepel', help, pos = 2, lib.loc = NULL)
library('ggplot2', help, pos = 2, lib.loc = NULL)

MakePlot = function (fin) {

  set.seed(1)

  numcol = ncol(fin)
  GOvec = as.matrix(fin[ , 2:(numcol-2) ])

  tsne <- Rtsne(GOvec, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)
  tsne_out = tsne$Y
  tsne_out = data.frame( cbind ( fin[,1], tsne_out, fin[,(numcol-1):numcol] ) )
  colnames(tsne_out) = c('name','dim1','dim2','ic','color1')

  this_plot = ggplot(tsne_out, aes(x = dim1, y = dim2, colour=factor(color1), size=100*ic)) +
    geom_point(alpha=.8, color='gray40' ) +
    theme_linedraw() + theme_light() +
    geom_point(alpha=.8, data=subset(tsne_out, color1==1 | color1==4) , aes(x = dim1, y = dim2, colour=factor(color1), size=100*ic ) ) +
    geom_text_repel(
      data = subset(tsne_out, color1==1 | color1==4),
      aes(label = name),
      size = 4,
      box.padding = unit(0.35, "lines"),
      point.padding = unit(0.3, "lines")
    ) +
    ggtitle (this_title) +
    labs(x="Dim 1", y="Dim 2") +
    theme(legend.position="left",plot.title = element_text(size=20,hjust = 0.5)) + 
    theme(axis.text=element_text(size=14),
          axis.title=element_text(size=16)) +
    scale_color_manual(values=c("mediumblue","firebrick1", "lavenderblush2", "lightcyan4", "gray92")) +
    guides(colour=FALSE, size=FALSE) #+
    # guides(size = guide_legend(override.aes = list(size=5)))
  return (this_plot)
}


setwd('C:/Users/dat/Documents/BertNotFtAARawSeqGO/mf/fold_1/2embPpiAnnotE256H1L12I512Set0/NoPpiNoTypeScaleFreezeBert12Ep10e10Drop0.1/checkpoint-60480')

# c/Users/dat/Documents/BertNotFtAARawSeqGO/mf/2embPpiAnnotE256H1L12I512Set0/YesPpiYesTypeScaleFreezeBert12Ep10e10Drop0.1

# GOvecFromModel
fin = read.csv("GOvecFromBert12.tsv",sep="\t",header=F,stringsAsFactors=F)
this_title = 'MF GO Vector Input'
plot1 = MakePlot(fin)

# fin = read.csv("GOvecFromModel.tsv",sep="\t",header=F,stringsAsFactors=F)
# this_title = 'GOvecFromModel'

fin2 = read.csv("../test_govec_hidden_layer.tsv",sep="\t",header=F,stringsAsFactors=F)
this_title = 'MF GO Vector Hidden Layer'
numcol = ncol(fin)
fin = cbind(fin2, fin[,(numcol-1):numcol] ) ## append ic and color
plot2 = MakePlot(fin)

ggsave('GOVectorInput.pdf',plot1,width=8,height=8,units='in')
ggsave('GOVectorHiddenLayer.pdf',plot2,width=8,height=8,units='in')

