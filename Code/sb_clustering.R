# get comfortable with tidyverse
# visualization practice
# clustering (k-means)


rm(list = ls())
# load necessary packages
library(tidyverse)
library(RColorBrewer)
library(dplyr)
library(tidyr)

# load data
sb = read.csv('Data/starbucks_drinkMenu_expanded.csv',
              stringsAsFactors = FALSE)


str(sb)

# these column names are awful - change them
colnames(sb) = c('Beverage_category', 'Beverage', 'Beverage_prep', 
                 'Calories', 'Fat', 'Trans_fat', 'Sat_fat', 'Sodium', 
                 'Carb', 'Chol', 'Fibre', 'Sugar', 'Protein', 'Vitamin_A',
                 'Vitamin_C', 'Calcium', 'Iron', 'Caffeine')
str(sb)

# --- cleaning fat ----

# ways to view data
View(sb)
unique(sb$Fat)

# view 3 2 column
sb[sb$Fat == '3 2',]
# this should definitely be 3.2
# don't change the original csv data

sb = sb %>%
  mutate(Fat = as.numeric(ifelse(Fat == '3 2', '3.2', Fat)))

summary(sb)

# ---- cleaning caffeine ----

unique(sb$Caffeine)

View(subset(sb, Caffeine %in% c("varies", "Varies", "")))
# caffeine has some odd words that can't be interpreted as numeric: varies and Varies
# note that tazo tea drinks really do have varying levels of caffeine

# for our purpose today, let's assume the tazo drinks are herbal, ie. 0 caffeine
# a google search leads up to believe 165 is appropriate for the iced coffee drink
# let's assume there is 0 caffeine in the chocolate banana smoothie

?str_detect
# this function gives a TRUE/FALSE depending on whether a 
# character string shows up in another character string

str_detect(c("Coffee", "coffee", "iced coffee", 'latte', 'COFFFEE'), 'co')

sb = sb %>%
  mutate(Caffeine = case_when(Caffeine %in% c("varies", "Varies", "") & str_detect(Beverage, "Tazo") ~ "0", 
                              Caffeine %in% c("varies", "Varies", "") & str_detect(Beverage, "Iced Brewed Coffee") ~ "165",
                              Caffeine %in% c("varies", "Varies", "") & str_detect(Beverage, "Banana Chocolate Smoothie") ~ "0",
                              TRUE ~ Caffeine))
summary(sb)

sb$Caffeine = as.numeric(sb$Caffeine)
summary(sb)

# ------ cleaning vitamins, calcium, iron ------
# each of these have the same problem: there are % symbols in each of the values
# use gsub to remove the % and replace it with a "" and then convert to numeric

sb = sb %>%
  mutate(
    Vitamin_A = gsub("%", "", Vitamin_A) %>% as.numeric(),
    Vitamin_C = gsub("%", "", Vitamin_C) %>% as.numeric(),
    Calcium = gsub("%", "", Calcium) %>% as.numeric(),
    Iron = gsub("%", "", Iron) %>% as.numeric()
  )
summary(sb)

# ---- k-means clustering -----

# k-means clustering (and hierarchical clustering) needs only numerical values as input
# thus, we need to remove some columns
# sb_X = subset(sb, select = -c(1, 2, 3))
# note that the above is not the best way
# it works this time with this arrangement of the data but what if thed data engineer rearranges
sb_X = subset(sb, select = -c(Beverage_category, Beverage, Beverage_prep))
summary(sb_X)

# units and scale matter very much in clustering

sb_stand = apply(sb_X, 2, function(x){(x - mean(x))/sd(x)}) # standardizing data (2 means columns, 1 would mean rows)
summary(sb_stand)

# now, time to do clustering on the standardized columns
max = 15 # this is the maximum number of clusters you think would be useful
wss = (nrow(sb_stand) - 1)*sum(apply(sb_stand, 2, var)) # what happens if there is just 1 cluster
for (i in 2:max){
  wss[i] = sum(kmeans(sb_stand, centers = i)$withinss)
}

ggplot() + 
  geom_line(aes(x = 1:max, wss)) + 
  geom_point(aes(x = 1:max, wss)) + 
  scale_x_continuous(breaks = c(1:max))


# 7 kind of looks like the elbow region of this plot
# in general, look for the area where the rate of decline
# starts to deteriorate

# fit your k-means algorithm

sb_kmeans = kmeans(sb_stand, centers = 7) # 7 is supported by the loop above

# clustering ins't useful until you explore how the clusters are different
# add clusters back to the original data: 

sb_X$km_cluster = as.factor(sb_kmeans$cluster)
View(sb_X)

head(sb_X)

# to visualize it is necessary to make this wide data
# into a long form

# using reshape2 to convert from wide to long form
# and vice versa

sb_long = melt(sb_X, id.vars = c("km_cluster"))
View(sb_long)

# how to visualize clusters of numeric variables:
# make a boxplot, for each cluster, of each numeric attribute/column

ggplot(data = sb_long) + 
  geom_boxplot(aes(x = km_cluster, y = value, fill = km_cluster)) + 
  facet_wrap(~ variable, scales = "free") + 
  scale_fill_brewer('Cluster \nDrinks', palette = "Dark2")

# typical to name clusters: 
# cluster 5: less healthy drinks
# cluster 1: high vitamin c
# cluster 4: healthy drinks

# add column into original data 
# to see what drink names are in each cluster
sb$km_cluster = as.factor(sb_kmeans$cluster)

sb[sb$km_cluster == "4",]











