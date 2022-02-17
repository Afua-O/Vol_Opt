# Libraries
library(ggplot2)
library(GGally)
library(dplyr)
library(hrbrthemes)
library(viridis)
library(plotly)
library(tidyverse)
library(parcoords)

#Set working directory (Remember to change from "\" to "/")
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions")
print(getwd())

# Add additional column with labels and then a row for max possible values and another for min possible values for each y-axis
data <- read.csv("hydro_max2.csv", header= TRUE)
head(data)

#####
ggparcoord(data=data,
           columns = c(1:4), groupColumn = 5, 
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = TRUE, 
           title = "Trade-offs (Standardized to Min = 0 and Max = 1)",
           alphaLines = 1
) + 
  scale_color_viridis(option = "b",discrete=TRUE) +
  theme_ipsum()+
  theme(
    legend.position="none",
    plot.title = element_text(size=10)
  ) +
  xlab("") +
  ylab("Direction of preference →" )
