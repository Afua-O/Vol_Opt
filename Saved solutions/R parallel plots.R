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

#data preparation#
#Add new column called "Name"
#For each row in the 'Name' column, put "Other solutions
# identify best solution for each objective and change the row from "Other solutions" to "Best x"
# add two rows, one with max possible value of each obj and the other with min possible (for standardization)
#add another row with the compromise(eg: the solution meeting target 80% of the target value)
#if 0 is the max/best/target solution, invert that variable
# Add additional column with labels and then a row for max possible values and another for min possible values for each y-axis

data <- read.csv("hydro_max_q2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best_hydropower", "Best_environment", "Best_flooding", "Best_irrigation", "Compromise", "Other solutions"))
 
ggparcoord(data=data,
           columns = c(1:4), groupColumn = 5,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "Trade-offs (Standardized to Min = 0 and Max = 1)",
           alphaLines = 0.5
) + geom_line(size=1)+
  #scale_color_manual(values = c(min, max, all_solutions, best_e-flows, best_hydro)) +
  scale_color_manual(values = c("red", "green", "blue", "orange", "grey", "grey", "grey")) +
  theme_ipsum()+
  theme(
    legend.position= "right",#"none",
    legend.title = element_blank(),
    plot.title = element_text(size=10)
  ) +
  xlab("") +
  ylab("Direction of preference ¡÷" )

######
