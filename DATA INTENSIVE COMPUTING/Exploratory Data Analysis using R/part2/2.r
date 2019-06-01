library(plotly)

#reading the file
TestedPositive<-read.csv("Prob2.csv", header=T)

Week_padded <- str_pad(Summary$WEEK,2,pad="0")
Week_padded

TestedPositive$NewColumn <- paste(TestedPositive$YEAR,Week_padded, sep='')

View(TestedPositive)



#plotting xaxis by week
xaxis <- c(TestedPositive$NewColumn)
xaxis

View(TestedPositive)
#assigning variables to each column

#yello
AsubtypeNotPerformed <- TestedPositive$A..Subtyping.not.Performed.
AsubtypeNotPerformed

#orange
AH1N1 <- TestedPositive$A..2009.H1N1.
AH1N1

#red
Ah3N2 <- TestedPositive$A..H3.
Ah3N2

#purple
H3n2v <- TestedPositive$H3N2v
H3n2v

#darkgreen
Blin <- TestedPositive$B
Blin

#florecent
Bviclin <- TestedPositive$BVic
Bviclin

#lightgreen
BYamlin <- TestedPositive$BYam
BYamlin

#keeping all columns in a dataframe
data <- data.frame(xaxis,AsubtypeNotPerformed,AH1N1,Ah3N2,H3n2v,Blin,Bviclin,BYamlin)
data

#plotting the columns in a bar



plotgraph2 <- plot_ly(data,x = c(xaxis)) %>%
  add_trace(y = ~BYamlin, type = 'bar', name = 'B(Yamagata Lineage)',marker = list(color = c('rgba(24,179,20,89)'))) %>%
  
  add_trace(y = ~Bviclin, type = 'bar', name = 'B(Victoria Lineage)',marker = list(color = c('rgba(59,229,54,67)'))) %>%
  
  add_trace(y = ~Blin, type = 'bar', name = 'B(Lineage Not Performed)',marker = list(color = c('rgba(17,86,15,100)'))) %>%
  
  add_trace(y = ~H3n2v, type = 'bar', name = 'H3N2v',marker = list(color = c('rgba(20,101,179,89)'))) %>%
  
  add_trace(y = ~Ah3N2, type = 'bar', name = 'H3N2v',marker = list(color = 'red')) %>%
  
  add_trace(y = ~AH1N1, type = 'bar', name = 'A (H1N1)pdm09',marker = list(color = 'orange')) %>%
  
  add_trace(y = ~AsubtypeNotPerformed, type = 'bar', name = 'A(Subtyping not performed)',marker = list(color = 'yellow')) %>%
  
    
layout(xaxis = list(type = 'category',title='Week',seq(201840,201918), tickangle=315,tick0=201840,tick10=201908,dtick=2,ticks = "outside",tickwidth = 2,showline=TRUE), 
       yaxis=list(seq(0,2500,by=500), title='Number of positive specimens',showline=TRUE),
       barmode = 'stack',title='Influenza Positive Tests Reported to CDC by Public Health Lab,
       National Summary, 2018-2019 Season<br><br>')

plotgraph2 






