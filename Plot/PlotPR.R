

library(tidyr)
library(ggplot2)
library(gridExtra)
library(ggrepel)
library(ggpubr)
library(directlabels)

fin = read.table("TransformerPredictGO.txt",header=T,stringsAsFactors=F)

method_show = c('DeepGOSeqFlat','DeepGOProtFlat','BlastEval10','BlastEval100','BaseTransformer','Motif','Motif+Deepgoppi','Motif+3D')

fin = subset( fin, fin$method %in% method_show )

PlotList = list()
metric_type = 'rec'

metricMap = list()
metricMap[['rec']] = 'Recall@k'
metricMap[['prec']] = 'Precision@k'

quantileMap = list()

freq="low"
ontotype="mf"

fin2 = subset(fin, fin$onto==ontotype & fin$frequency==freq ) 
data_long = gather(fin2, Kvalue, Value, X10:X50, factor_key=TRUE)
data_long$Kvalue = as.numeric(data_long$Kvalue) * 10 

data_rec = subset(data_long,metric=='rec')
p <- ggplot(data=data_rec,aes( x = Kvalue, y = Value, group=factor(method),color=factor(method)) )
p <- p + geom_line( data=data_rec,aes( x = Kvalue, y = Value, group=factor(method),color=factor(method), size=1.1 ) ) + 
geom_dl(aes(label = method), method = list(dl.trans(x = x + 0.2), "last.points", cex = 1.5, vjust=.5)) +
geom_dl(aes(label = method), method = list(dl.trans(x = x - 0.2), "first.points", cex = 1.5, vjust=.5)) +
ggtitle(paste (toupper(ontotype),"Low Frequency") )
scale_y_continuous( name='Recall') +
scale_x_continuous( name="K", breaks=seq(10,50,10), labels=seq(10,50,10), limits=c(0,60) ) +
# guides(colour = guide_legend(override.aes = list(size=5))) +
guides(size=FALSE, colour=FALSE) + theme_linedraw() + theme_light()
# theme(legend.position="top",plot.title = element_text(hjust = 0.5)) 
p


# adding the relative humidity data, transformed to match roughly the range of the temperature
# data_prec = subset(data_long,metric=='prec')
# p <- p + geom_line(data=data_prec,aes(x = Kvalue,y = Value*10, group=factor(method),color=factor(method)))
# # now adding the secondary axis, following the example in the help file ?scale_y_continuous
# # and, very important, reverting the above transformation
# p <- p + scale_y_continuous(sec.axis = sec_axis(~./10, name = "Precision"))


for (frequency in c( "low","middle","high" ) ) {

  for (ontotype in c("bp","mf","cc")) {

    fin2 = subset(fin,fin$ontology==ontotype & fin$metric==metric_type & fin$Method%in%Method2get & quantile==frequency)

    data_long = gather(fin2, f1threshold, f1value, X10:X50, factor_key=TRUE)

    PlotList[[ paste0(ontotype,frequency) ]] = ggplot(data=data_long, aes(x=f1threshold, y=f1value, group=factor(name) )  ) +
    geom_line(aes(color=factor(name),linetype=factor(name)),size=1.1,alpha=0.8) +
    scale_x_discrete( name="k", labels=seq(10,25,5) ) +
    scale_y_continuous( name=paste0(metricMap[[metric_type]])) + # + ylim(0, .65) + # , limits=c(0,.65)
    scale_colour_manual(name = "", labels = label_name, values=cbbPalette ) + # palette = 'Set1',direction=-1, type = 'div'
    theme(legend.position="left",plot.title = element_text(hjust = 0.5)) +
    ggtitle(paste (toupper(ontotype),quantileMap[[frequency]]) ) + theme_linedraw() + theme_light() +
    guides(colour = guide_legend(override.aes = list(size=5))) +
    guides(linetype = FALSE) +
    theme(legend.text=element_text(size=16))

  }

}


ggarrange(
PlotList[['bpall']], PlotList[['bpquant25']], PlotList[['bpbetweenQ25Q75']], PlotList[['bpquant75']],
PlotList[['mfall']], PlotList[['mfquant25']], PlotList[['mfbetweenQ25Q75']], PlotList[['mfquant75']],
PlotList[['ccall']], PlotList[['ccquant25']], PlotList[['ccbetweenQ25Q75']], PlotList[['ccquant75']],  common.legend = TRUE, ncol=4, nrow=3)

