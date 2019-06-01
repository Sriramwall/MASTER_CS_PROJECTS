library(plotly)

#reading the data from CSV files of Corresponding years

#Illness 2018-2019
Data1<-read.csv("ILINet_2018_2019.csv", header=T, skip=1) 
Data1

View(Data1)
Week_padded <- str_pad(Data2$WEEK,2,pad="0")
Week_padded
Data2$WEEK
xaxis <- c(Data2$WEEK)
xaxis
length(xaxis)
length(Data7$X..WEIGHTED.ILI)

#Illness 2017-2018
Data2<-read.csv("ILINet_2017_2018.csv", header=T,skip=1) 
Data2

#Illness 2016-2017
Data3<-read.csv("ILINet_2016_2017.csv", header=T,skip=1) 
Data3

#Illness 2015-2016
Data4<-read.csv("ILINet_2015_2016.csv", header=T,skip=1)
Data4

#Illness 2014-2015
Data5<-read.csv("ILINet_2014_2015.csv", header=T,skip=1)
Data5

#Illness 2011-2012
Data6<-read.csv("ILINet_2011_2012.csv", header=T,skip=1)
Data6

#Illness 2009-2010
Data7<-read.csv("ILINet_2009_2010.csv", header=T,skip=1)
Data7

#Creating a new column in first data set of Year 2018-2019
Data2$NewColumn <- paste(Data2$YEAR, Data1$WEEK, sep='')
Data2

Data2$X..WEIGHTED.ILI

View(Data2)


#creating a data frame 
data<-data.frame(Data2$X..WEIGHTED.ILI)

plotgraph4<-plot_ly(data,x = c(xaxis)) %>%
  add_trace(y = ~c(Data2$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2017-18 Season', 
            line = list(color='yellow')) %>%
  
  add_trace(y = ~c(Data3$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2016-17 Season', 
            line = list(color='blue')) %>%
  
  add_trace(y = ~c(Data4$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2015-16 Season', 
            line = list(color='orange')) %>%
  
  add_trace(y = ~c(Data5$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2014-15 Season', 
            line = list(color='pink')) %>%
  
  add_trace(y = ~c(Data6$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2011-12 Season', 
            line = list(color='green')) %>%
  
  add_trace(y = ~c(Data7$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines', name = '2009-10 Season', 
            line = list(color='grey')) %>%
  
  add_trace(y = ~c(Data1$X..WEIGHTED.ILI), type = 'scatter', mode = 'lines+markers', name = '2018-19 Season', 
            line = list(color='red')) %>%
  
  add_trace(y = ~c(rep(2.2,times = 52)), type = 'scatter', mode = 'lines', name = 'National Baseline', 
            line = list(color='black', dash = 'dash')) %>%
  
layout(xaxis = list(range = c(0,60),title = "Week", type = "category",tick0=40,dtick=2,showline=TRUE),
yaxis = list(seq(0,8,by=1),title = '% of Visits for ILI',showline=TRUE),
         title='Percentage of visits for Influenza-like Illines(ILI) Reported by\nthe U.S. Outpatient Influenza-like Illness Surveillance Network (ILINet),\nWeekly National Summary, 2018-2019 and Selected Previous Seasons')

plotgraph4
