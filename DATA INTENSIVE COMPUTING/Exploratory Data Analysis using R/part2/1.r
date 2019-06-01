library(plotly)
library(stringr)

# input file : Sum.csv
Summary<-read.csv("Sum.csv", header=T)

Week_padded <- str_pad(Summary$WEEK,2,pad="0")
Week_padded

Summary$NewColumn <- paste(Summary$YEAR,Week_padded, sep='')



View(Summary)
Summary$NewColumn
#Keeping X-axis as NewColumn created in the previous line (Year and Week combined)
xaxis <- c(Summary$NewColumn)

#Plotting y axis with Influenza A and InFluenzaB
totalA <- Summary$TOTAL.A
y1
totalB <- Summary$TOTAL.B
y2

#PLotting a scattered line of PercentPositive Column
percentPositive <- Summary$PERCENT.POSITIVE
percentPositive
#forming the data frame with Xaxis and y1 and y2
data <- data.frame(xaxis,y1,y2)
data


#plotting the graoh
plotgraph1 <- plot_ly(data, x = c(xaxis)) %>%
  
  add_trace(y = ~totalA, type = 'bar', name = 'A',marker = list(color = 'yellow')) %>%
  
  add_trace(y = ~totalB, type = 'bar',name = 'B',marker = list(color = 'green')) %>%
  
  add_trace(y = ~percentPositive, type = 'scatter', mode = 'lines', name = 'Percent Positive',yaxis = 'y2',
            line = list(color='black')) %>%
  
  add_trace(y = ~c(Summary$PERCENT.A), type = 'scatter', mode = 'lines', name = 'Percent Positive',
            yaxis = 'y2',line = list(color='yellow', dash = 'dot',side='right')) %>%
  
  add_trace(y = ~c(Summary$PERCENT.B), type = 'scatter', mode = 'lines', name = 'Percent Positive',
            yaxis = 'y2',line = list(color='green', dash = 'dot')) %>%
  
  layout(xaxis = list(type = 'category',title = "Week",tickangle=315,tick0=201840,tick10=201908,dtick=2,ticks = "outside",tickwidth = 2,showgrid=TRUE,showline=TRUE),yaxis = list(seq(0,18000,by=2000),title = 'Number of positive specimens',showgrid=TRUE,showline=TRUE),
         yaxis2 = list(range = c(0,30),overlaying = 'y',side = 'right', title = "Percent Positive",showgrid=TRUE,showline=TRUE), 
         barmode = 'overlay', title='Influenza Positive Tests Reported to CDC by U.S. Clinical Lab,
         National Summary, 2018-2019 Season')
plotgraph1


