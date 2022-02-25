# Libraries
library(ggplot2)
library(GGally)
library(magrittr)
library(dplyr)
library(hrbrthemes)
library(viridis)
library(plotly)
library(tidyverse)
library(parcoords)

#Set working directory (Remember to change from "\" to "/")
setwd("C:/Users/aow001/Dropbox/PhD/Work/4. Hydro-economic modelling/Python optimisation/Vol_Opt/Saved solutions")
print(getwd())

# import data and convert to tibble
data <- read.csv("hydro_max_q2.csv", header= TRUE)
head(data)
solution <- as_tibble(data)
solution

#calculate axis breaks and set coordinates of tick marks
axis_df <- stack(solution[-5]) %>%
            group_by(ind) %>%
            summarize(breaks = pretty(Values, n = 5),
                      yval = (breaks - min(breaks))/(max(values) - min(values))) %>%
            mutate(xmin = as.numeric(ind) - 0.05,
                  xmax = as.numeric(ind),
                  x_text = as.numeric(ind) - 0.2)

#calculate co-ordinates for actual axis lines
axis_line_df <- axis_df %>% 
                group_by(ind) %>%
                summarize(min = min(yval), max = max(yval))

#reshape and normalize original data
lines_df <- solution[-5] %>%
  mutate(across(everything(), function(x) (x - min(x))/(max(x)))) %>%
  stack() %>%
  mutate(row = rep (solution$Name, ncol(solution) -1))

#Plot
ggplot(lines_df, aes(ind, values, group = row)) +
  geom_line(color =  "orange", alphas = 0.5) +
  geom_segment(data = axis_line_df, aes(x = ind, xend = ind, y = min, yend = max),
               inherit.aes = FALSE) +
  geom_segment(data = axis_df, aes(x = xmin, xend = xmax, y = yval, yend = yval),
               inherit.aes = FALSE) +
  geom_text(data = axis_df, aes(x = x_text, y = yval, label = breaks),
            inherit.aes = FALSE) +
  geom_text(data = axis_line_df, aes(x = ind, y = 1.2, label = ind),
            size = 6, inherit.aes = FALSE, check_overlap = TRUE, hjust = 1) +
  theme_void() +
  theme(plot.margin = margin(50, 20, 50, 20))
###################




  
    