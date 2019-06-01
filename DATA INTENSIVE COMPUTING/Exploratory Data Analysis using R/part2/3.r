library(plotly)

# reading data from Prob3.csv
Mortality<-read.csv("Prob3_check.csv", header=T) 

#Viewing the data
Mortality
View(Mortality)

Week_padded <- str_pad(Mortality$Week,2,pad="0")
Week_padded


#creating a new column in csv file
Mortality$NewColumn <- paste(Mortality$Year,Week_padded, sep='')
Mortality$NewColumn

#creating data frame with required column headers
data<-data.frame(Mortality$Week,Mortality$Threshold,Mortality$Expected,Mortality$Percent.of.Deaths.Due.to.Pneumonia.and.Influenza)
data


Mortality$Threshold
Mortality$Expected
#Plotting the graph
plotgraph3 <-plot_ly(data, x = ~Mortality$NewColumn) %>%
  add_trace(y = ~c(Mortality$Threshold),type = 'scatter', mode = 'lines', 
  line = list(color='black',width =2)) %>%
 
   add_trace(y = ~c(Mortality$Expected), type = 'scatter', mode = 'lines',
            line = list(color='black')) %>%
  
  add_trace(y = ~c(Mortality$Percent.of.Deaths.Due.to.Pneumonia.and.Influenza), type = 'scatter', 
            mode = 'lines',line = list(color='red')) %>%
  
  layout(xaxis = list(title = "Week", type = "category",categoryarray=c(0,432),showline=TRUE), 
         
         yaxis = list(range = c(4,12),title = '% of All Deaths due to P&I',showline=TRUE),
         
         title='Pneumonia and Influenza Mortality from\nthe National Center for Health Statics Mortality Surveillance System
         Data through the week ending February 9, 2019, as of February 21,2019')

plotgraph3

