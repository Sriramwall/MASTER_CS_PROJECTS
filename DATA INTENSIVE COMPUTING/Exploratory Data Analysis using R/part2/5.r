library(plotly)

#reading the heap map data from file
HeatMap <- read.csv("FluHeatMap_USA.csv", header=T) # input file : 'FluHeatMap_USA.csv'

p <- plot_geo(HeatMap, locationmode = 'USA-states') %>%
  add_trace(
    z = ~HeatMap$LEVEL, locations = ~HeatMap$STATE,
    color = ~HeatMap$LEVEL, colors = 'Reds' ) %>%
  colorbar(title = "ILI Activity Level") %>%
  layout(
    title = '2017-18 Influenza Season Week 4 ending Jan 27, 2018',
    geo = list(
      scope = 'usa',
      projection = list(type = 'albers usa'),
      showlakes = TRUE,
      lakecolor = toRGB('white')
    )
  )

p