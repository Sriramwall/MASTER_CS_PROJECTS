library(shiny)
library(plyr)
library(plotly)
library(ggmap)
library(dygraphs)

read <- read.csv("master_state_count.csv", header=F)
state_code <- state.abb[match(read$V1,state.name)]
count_state <- count(state_code)  
l <- list(color = toRGB("white"), width = 2)
g <- list(
  scope = 'usa',
  projection = list(type = 'albers usa'),
  showlakes = TRUE,
  lakecolor = toRGB('white')
)
twitterAll <- plot_geo(count_state, locationmode = 'USA-states') %>%
  add_trace(
    z = ~count_state$freq, locations = ~count_state$x,
    color = ~count_state$freq, colors=c("dark green","green", "yellow", "orange", "red")
  ) %>%
  colorbar(title = "Frequency of Tweets") %>%
  layout(
    title = '2018-19 Frequency of Tweets (All key words) on Flu Categorized by states',
    geo = g
  )



HeatMap <- read.csv("StateDataforMap_2018-19week8.csv", header=T) # input file : 'FluHeatMap_USA.csv'

HeatMap$ACTIVITY.LEVEL <- as.integer(gsub('[a-zA-Z]', '', HeatMap$ACTIVITY.LEVEL))

HeatMap$STATENAME <- state.abb[match(HeatMap$STATENAME,state.name)]
cdc <- plot_geo(HeatMap, locationmode = 'USA-states') %>%
  add_trace(
    z = ~HeatMap$ACTIVITY.LEVEL, locations = ~HeatMap$STATENAME,
    color = ~HeatMap$ACTIVITY.LEVEL, colors=c("dark green","green", "yellow", "orange", "red") ) %>%
  colorbar(title = "ILI Activity Level") %>%
  layout(
    title = '2018-19 Influenza Season Week 1 March 08, 2019',
    geo = list(
      scope = 'usa',
      projection = list(type = 'albers usa'),
      showlakes = TRUE,
      lakecolor = toRGB('white')
    )
  )

read <- read.csv("master_state_count_fewkey.csv", header=F)
state_code <- state.abb[match(read$V1,state.name)]
count_state <- count(state_code)  
twitterFew <- plot_geo(count_state, locationmode = 'USA-states') %>%
  add_trace(
    z = ~count_state$freq, locations = ~count_state$x,
    color = ~count_state$freq, colors=c("dark green","green", "light yellow", "orange", "red")
  ) %>%
  colorbar(title = "Frequency of Tweets") %>%
  layout(
    title = '2018-19 Frequency of Tweets (Few key words) on Flu Categorized by states',
    geo = g
  )
ui <- fluidPage(

  headerPanel("Heat map comparison"),

  sidebarPanel(
    selectInput("map", "Maps:", 
                c("CDC vs Twitter All" = "cta",
                  "Twitter All vs Twitter Few" = "tatf",
                  "CDC vs Twitter Few" = "ctf"))),

  mainPanel(plotlyOutput("graph1"), plotlyOutput("graph2"))

)



server <- function(input, output) {
  output$graph1 <- renderPlotly({
    
    if (input$map=='tatf') {
      twitterAll;
    } else {
      cdc;
    }
    
  })
  output$graph2 <- renderPlotly({
    if (input$map=='cta') {
      twitterAll;
    } else {
      twitterFew;
    }
  })
}


shinyApp(ui, server)
