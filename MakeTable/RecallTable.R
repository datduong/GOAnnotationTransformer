
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)

fin2 = read.table('GOannotationRecallJan21.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100

#! remove some for table in powerpoint 
fin2[,c(2,3,5,6,8,9)] = NULL 

# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP"=1, "MF"=1, "CC"=1)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BLAST Psi-BLAST", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGO", 3, 4, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Transformer", 5, 8, label_row_css = "background-color: #666; color: #fff;")

#### recall rate on rare labels for LARGER DATA.
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)
fin2 = read.table('GOannotationRecallJan21Large.txt',sep='\t',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP"=3, "MF"=3, "CC"=3)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BLAST Psi-BLAST", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGO", 3, 4, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Transformer", 5, 8, label_row_css = "background-color: #666; color: #fff;")



#! recall rate on rare labels for LARGER DATA. with AUC
library(taRifx)
library(xtable)
library(knitr)
library(kableExtra)
# fin2 = read.table('GOannotationRecallJul11LargeRareLabel.txt',sep='\t',header=T,stringsAsFactors=F)
fin2 = read.table('GOannotationRecallJul11LargeCommonLabel.txt',header=T,stringsAsFactors=F)
fin2[,2:ncol(fin2)] = fin2[,2:ncol(fin2)] # * 100
# fin2$Row = 1:nrow(fin2)
# colnames(fin2)=NULL
kable( fin2, "latex", longtable = F, booktabs = T, caption = "blank", row.names=TRUE ) %>% #row.names=c(1:nrow(fin2))
add_header_above(c(" "=2, "BP"=4, "MF"=4, "CC"=4)) %>%
kable_styling(latex_options = c("hold_position","scale_down"), position = "center") %>%
pack_rows("BLAST Psi-BLAST", 1, 2, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGO", 3, 4, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("DeepGOPlus", 5, 6, label_row_css = "background-color: #666; color: #fff;") %>% 
pack_rows("Transformer", 7, 9, label_row_css = "background-color: #666; color: #fff;")

