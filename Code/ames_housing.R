rm(list = ls())

library(ggplot2)
library(pROC)
library(randomForest)
library(rpart)
# install.packages("logistf")
library(logistf) # if you need firth's penalized likelihood
library(rpart.plot)
library(tidyverse)

house = read.csv("Data/AmesHousing_sub_luxury.csv", stringsAsFactors = TRUE)

str(house)

# note: PID is a unique home identifier, we need to remove it before
# training our model

summary(house)

# note: Garage.Finish has a missing value (though R doesn't recognize 
# it as such). we can either 1) remove that row or 2) impute it with
# a low influence value, such as the mode

# Kitchen.Quality has just one observation of "Po" (poor kitchen)
# this will present a problem if that observation shows up in the testing 
# data and not in the training data
# it is typical in this kind of situation to combine levels 
# our plan: combine "Po" and "TA" levels

# while we're at it: let's change the levels to be more understandable
# to the average reader

# clean kitchen quality
# factors are hard to change... make it a character first
house$Kitchen.Qual = as.character(house$Kitchen.Qual)
# change the value of "Po" and "TA" to be instead "Po/TA"
house$Kitchen.Qual[house$Kitchen.Qual %in% c("Po", "TA")] = "Po/TA"
# did we do this right
table(house$Kitchen.Qual)

# make it back into a factor, and at the same time, make the labels better for the reader

house$Kitchen.Qual = factor(house$Kitchen.Qual,
                            levels = c("Po/TA", "Fa", "Gd", "Ex"),
                            labels = c("Poor/Typical/Average", "Fair", "Good", "Excellent"))

summary(house)
str(house)

# clean garage finish
# garage finish has a missing value, though it is stated as " " instead of NA
# thus, we can't use the function is.na() like we did before
# plan: impute all " " values with the mode ("Unf")

house$Garage.Finish = as.character(house$Garage.Finish)
table(house$Garage.Finish)
house$Garage.Finish[house$Garage.Finish == ""] = "Unf"
# again, make it into a factor but with ordered meaningful levels
house$Garage.Finish = factor(house$Garage.Finish,
                             levels = c("Unf", "RFn", "Fin"),
                             labels = c("Unfinished", "Rough Finished", "Finished"))

RNGkind(sample.kind = "default")
set.seed(4561)
train.idx = sample(x = 1:nrow(house), size = .7*nrow(house))
train.df = house[train.idx, ]

# remove PID
train.df = train.df %>%
  subset(select = -c(PID))

# or
# train.df = subset(train.df, select = -c(PID))

test.df = house[-train.idx, ]

# the column luxury is what you want to predict. note that column PID is a home
# identifier and should not be used in the algorithm. build a tree that would be 
# suitable for making predictions on new data

# making tree
ctree = rpart(luxury ~ .,
              data = train.df, 
              method = "class")
# plotting tree
rpart.plot(ctree)
# looks from initial tree garage cars is the most important factor in 
# dictating whether a house is luxury or not

# pruning tree
optimalcp = ctree$cptable[which.min(ctree$cptable[,"xerror"]),"CP"]
ctree2 = prune(ctree, cp = optimalcp)
# plotting pruned tree
rpart.plot(ctree2)

test.df$result_pred = predict(ctree2, test.df, type = "class")

# make confusion matrix
table(test.df$result_pred, test.df$luxury)

pi_hat = predict(ctree2, test.df, type = "prob")[, "Luxury"]

# make an ROC curve with the final tree. what is your AUC. interpret 
# sensitivity and specificity

rocCurve = roc(response = test.df$luxury, # supply truth (in test set)
               predictor = pi_hat, # supply predicted PROBABILITIES of positive case
               levels = c("Standard", "Luxury") # (negative, positive)
)

plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# what can we say from this?
# AUC = 0.891
# pi* = 0.206 
# we will only predict a house is luxury if the probability is greater than 0.206
# sensitivity: we correctly predict a house is luxury about 
# 94.4% of the time when the house is actually luxury
# specificity: we correctly predict a house being standard about
# 78.9% of the time when it actually is standard


# save a column of the categorical predictions for the test data. these predictions 
# should be such that the resulting sensitivity and specificity are as advertised in 
# the previous question

# save our predictions in our test set
pi_star = coords(rocCurve, "best", ret = "threshold")$threshold[1]

# create a column of predicted values for test set
test.df$luxury_pred = ifelse(pi_hat > pi_star, "Luxury", "Standard")

# interpret/state at least two of the classification rules

# build a random forest that would be suitable for making predictions
# on new data. for tuning, assume 500 trees is sufficient but be sure to tune m

myforest = randomForest(luxury ~ .,
                        data = train.df,
                        ntree = 500,
                        mtry = 4)







