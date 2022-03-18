# Libraries
library(ggplot2)
library(GGally)
library(dplyr)
library(hrbrthemes)
library(viridis)
library(plotly)
library(tidyverse)
library(parcoords)

#########
##Best for each user

#Set working directory (Remember to change from "\" to "/")
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/1984_clam_stochastic95")
      #29y_clam/seed10")
print(getwd())

#data preparation#
#Add new column called "Name"
#For each row in the 'Name' column, put "Other solutions"
# identify best solution for each objective and change the row from "Other solutions" to "Best x"
# add two rows, one with max possible value of each obj and the other with min possible (for standardization)
#add another row with the compromise(eg: the solution meeting target 80% of the target value)
#if 0 is the max/best/target solution, invert that variable so min can be -Max
# Move all the "Best_X" rows to the last rows of the csv so they lay above the "other solutions" in the plot

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Other solutions", "Min", "Max"))
 
ggparcoord(data=data,
           columns = c(2,1,3,4), groupColumn = 5,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "Trade-offs: clam e-flows (Standardized to Min = 0 and Max = 1)",
           alphaLines = 0.3
) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#00CC99", "#6600FF", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "right",#"none",
    legend.title = element_blank(),
    plot.title = element_text(size=10)
  ) +
  xlab("") +
  ylab("Direction of preference ->") +
  theme(axis.title.y = element_text(hjust=0.95, size = 11)) 

##########
###See if there is compromise- 80%
#Save the solutions file as a new file
#Highlight 80% e-flows threshold an 80th percentile hydropower
#change the Name column into 3 categories: "Eflow" for the solutions meeting 80% for the e-flows column, 'Hydropower' for the solutions above 80th percentile of the hydropower and leave the rest as Other solutions for the res


data <- read.csv("10_solution3.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Hydropower", "Eflow", "Other solutions", "Min", "Max"))

ggparcoord(data=data,
           columns = c(4,1,3,2), groupColumn = 5,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "Trade-offs: clam e-flows (Standardized to Min = 0 and Max = 1)",
           alphaLines = 0.2
) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#33CCCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "right",#"none",
    legend.title = element_blank(),
    plot.title = element_text(size=10)
  ) +
  xlab("") +
  ylab("Direction of preference ???" ) +
  theme(axis.title.y = element_text(hjust=0.95, size = 11)) 
