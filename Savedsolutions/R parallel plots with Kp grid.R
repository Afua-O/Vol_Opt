# Libraries
library(ggplot2)
library(GGally)
library(dplyr)
library(hrbrthemes)
library(viridis)
library(plotly)
library(tidyverse)
library(parcoords)
library(gridExtra)
library(grid)
library(gtable)

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
#############################################################

#S1
#Set working directory (Remember to change from "\" to "/")
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc1")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair environment", "Other solutions", "Min", "Max"))

clam1<-
  ggparcoord(data=data,
             columns = c(2,5,1,3,4), groupColumn = 6,     
             scale= "uniminmax", # for no scaling use "globalminmax"
             showPoints = FALSE, 
             title = "Clam e-flows",
             alphaLines = 0.3
  ) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "none",
    plot.title = element_text(size=10, hjust = 0.5),
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13),
    plot.margin=unit(c(0,0.5,0,0),"cm")
  ) +
  xlab("") +
  ylab("Scenario 1")  + #ylab("") for eflows2 & 3
  theme(axis.title.y = element_text(hjust=0.5, size = 10, face = "bold")) 
##################################################

#S2
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc2")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair hydropower", "Fair environment", "Other solutions", "Min", "Max"))

clam2<- 
ggparcoord(data=data,
           columns = c(2,5,1,3,4), groupColumn = 6,     
           scale= "uniminmax", # for no scaling use "globalminmax"
           showPoints = FALSE, 
           title = "",
           alphaLines = 0.3
) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#FFCCFF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "none",
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13),
    plot.margin=unit(c(0.5,0.5,0,0),"cm")
  ) +
  xlab("") +
  ylab("Scenario 2")  +
  theme(axis.title.y = element_text(hjust=0.5, size = 10, face = "bold"))  
##################################################

#S3
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc3")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair hydropower", "Fair environment", "Other solutions", "Min", "Max"))

clam3<- 
  ggparcoord(data=data,
             columns = c(2,5,1,3,4), groupColumn = 6,     
             scale= "uniminmax", # for no scaling use "globalminmax"
             showPoints = FALSE, 
             title = "",
             alphaLines = 0.3
  ) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#FFCCFF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "none",
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13),
    plot.margin=unit(c(0.5,0.5,0,0),"cm")
  ) +
  xlab("") +
  ylab("Scenario 3")  +
  theme(axis.title.y = element_text(hjust=0.5, size = 10, face = "bold")) 
##################################################

#S4
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc4")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair hydropower", "Fair environment", "Other solutions", "Min", "Max"))

clam4<- 
  ggparcoord(data=data,
             columns = c(2,5,1,3,4), groupColumn = 6,     
             scale= "uniminmax", # for no scaling use "globalminmax"
             showPoints = FALSE, 
             title = "",
             alphaLines = 0.3
  ) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#FFCCFF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "none",
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13),
    plot.margin=unit(c(0.5,0.5,0,0),"cm")
  ) +
  xlab("") +
  ylab("Scenario 4")  +
  theme(axis.title.y = element_text(hjust=0.5, size = 10, face = "bold"))  
##################################################

#S5
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions/clam_cc5")
print(getwd())

data <- read.csv("10_solution2.csv", header= TRUE)
head(data)

#specify the preferred legend order
data$Name <- factor(data$Name, levels = c("Best hydropower", "Best environment", "Best irrigation", "Fair hydropower", "Fair environment", "Other solutions", "Min", "Max"))

clam5<- 
  ggparcoord(data=data,
             columns = c(2,5,1,3,4), groupColumn = 6,     
             scale= "uniminmax", # for no scaling use "globalminmax"
             showPoints = FALSE, 
             title = "",
             alphaLines = 0.3
  ) + geom_line(size=1)+
  scale_color_manual(values = c("#CC3366", "#006600", "#6600FF","#FFCCFF","#CCFFCC", "#E8E8E8", "#E8E8E8", "#E8E8E8")) +
  theme_ipsum()+
  theme(
    legend.position= "none",
    panel.grid.major.x =element_line(colour="black"),
    axis.text = element_text( colour = "black"),
    text = element_text(size = 13),
    plot.margin=unit(c(0.5,0.5,0,0),"cm")
  ) +
  xlab("") +
  ylab("Scenario 5")  +
  theme(axis.title.y = element_text(hjust=0.5, size = 110, face = "bold"))  
##################################################



#############################

grid.arrange(
  clam1,clam1,clam1,#eflow2_1, eflow3_1
  clam2,clam2,clam2,
  clam3,clam3,clam3,
  clam4,clam4,clam4,
  clam4,clam4,clam4,#clam5,
  nrow = 5,
  #top = "Future Climate Scenarios",
  left = textGrob(
    "Direction of preference ???",
    gp = gpar(fontface = 3, fontsize = 13),
    hjust = 0.5,
    x = 0.5,
    rot = 90
  )
)