# Libraries
library(ggplot2)
library(GGally)
library(dplyr)
library(hrbrthemes)
library(viridis)
library(plotly)
library(tidyverse)
library(parcoords)

#data preparation#
#Add new column called "Name"
#For each row in the 'Name' column, put "Other solutions"
# identify best solution for each objective and change the row from "Other solutions" to "Best x"
# add two rows, one with max possible value of each obj and the other with min possible (for standardization)
#
#Change the Name column into 5 categories: "Fair environment" for the solutions meeting 80% for the e-flows column, 'Fair hydropower' 
#for the solutions above 80th percentile of the hydropower and leave the rest as Other solutions for the res
#if 0 is the max/best/target solution, invert that variable so min can be -Max
#Label the rest as "Other solutions"
# Move all the "Best_X" rows to the last rows of the csv so they lay above the "other solutions" in the plot


#Set working directory (Remember to change from "\" to "/")
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc2")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair hydropower", "Fair environment", "Other solutions", "Min", "Max"))
 
ggparcoord(data=data,
           columns = c(2,5,1,3,4), groupColumn = 6,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "Clam cc2",
           alphaLines = 0.3
) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#FFCCFF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "bottom",#"none",
    legend.title = element_blank(),
    plot.title = element_text(size=10),
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13)
  ) +
  xlab("") +
  ylab("Direction of preference ???") +
  theme(axis.title.y = element_text(hjust=0.95, size = 13)) 


################################################
#When one group is missing (Eg: fair hydropower in the cc1 projections)

setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc1")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair environment", "Other solutions", "Min", "Max"))

ggparcoord(data=data,
           columns = c(2,5,1,3,4), groupColumn = 6,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "Clam cc1",
           alphaLines = 0.3
) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "bottom",#"none",
    legend.title = element_blank(),
    plot.title = element_text(size=10),
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13)
  ) +
  xlab("") +
  ylab("Direction of preference ???") +
  theme(axis.title.y = element_text(hjust=0.95, size = 13)) 