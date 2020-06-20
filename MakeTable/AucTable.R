

library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOannotationJan21.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP"=3, "MF"=3, "CC"=3)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BLAST Psi-BLAST", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGO", 3, 4, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Transformer", 5, 8, label_row_css = "background-color: #666; color: #fff;")


%>% 

add_indent( c(5:10)) %>%
add_indent( c(12:14)) %>%





####



library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOannotationMay15.txt',header=F,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 1
temp = fin2[,2]
fin2[,2] = fin2[,3]
fin2[,3] = temp ## because we want BP MF CC
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE, digits=3 ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "Fmax"=3)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") 



