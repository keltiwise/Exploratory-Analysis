
# clear your R environment prior to each new analysis
rm(list = ls())

# install packages (if needed)
# note: only need to install packages ONCE 
# install.packages("rpart") # used for fitting classification trees
# install.packages("rpart.plot") # for plotting treets
# install.packages("ggplot2") # for high qualiry data visualizations
# install.packages("pROC") # ROC curves

# load packages (do at beginning of every session)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(pROC)

# read in data

# --------- Introduction -------------

# read in data using relative file path
cdata = read.csv("Data/tennis.csv", stringsAsFactors = TRUE)
# do not attach your data

# look at column names, first few rows
head(cdata)
str(cdata)
summary(cdata)

# ----------- Exploratory Analysis -----------

# univariate plots of x and y variables
# 1st serve % in (X1stIn) (numeric) --> histogram

ggplot(data = cdata) + 
  geom_histogram(aes(x = X1stIn))

# ace rate (A.) (numeric) --> histogram
ggplot(data = cdata) +
  geom_histogram(aes(x = A.))

# surface (character) --> barchart
ggplot(data = cdata) + 
  geom_bar(aes(x = Surface))

# how many times did she play on carpet?
table(cdata$Surface)
# gives us precise numbers

# look at our Y variable, whihc is discrete
ggplot(data = cdata) + 
  geom_bar(aes(x = Result))

# clearly shes's won a lot, but what affects her winning probability?

# multivariate plots of x variables and y variable

# relationship between X1stIn and Result
ggplot(data = cdata) + 
  geom_point(aes(x = X1stIn, y = Result))
# this isn't always a great way to display data

ggplot(data = cdata) + 
  geom_histogram(aes(x = X1stIn, fill = Result))

ggplot(data = cdata) + 
  geom_histogram(aes(x = X1stIn, fill = Result), position = "fill")

# do something similar with Surface

ggplot(data = cdata) + 
  geom_point(aes(x = Surface, y = Result))

# this is clearly bad

ggplot(data = cdata) + 
  geom_bar(aes(x = Surface, fill = Result), position = "fill")

# serena's performance over time, but let's polish it this time

ggplot(data = cdata) + 
  geom_histogram(aes(x = year, fill = Result), binwidth = 1, position = "fill") + 
  labs(x = "Year", y = "Proportion") + 
  ggtitle("Serena's Wins Over Time") + 
  scale_fill_grey("Serena's \nOutcome")

# your turn do the same for ace rate - remember to play with binwidth
ggplot(data = cdata) + 
  geom_histogram(aes(x = 100*A., fill = Result), binwidth = 5, position = "fill") + 
  labs(x = "Ace Rate(%)", y = "Proportion") + 
  ggtitle("Serena's Win's Over Time") +
  scale_fill_grey("Serena's \nOutcome")

# two categorical x variables PLUS a categorical y variable
# ex: how do surface and higher rank affect winning

ggplot(data = cdata) + 
  geom_bar(aes(x = Surface, fill = Result), position = "fill") + 
  facet_wrap(~Higher.Rank.)
# this plot is good, but still is not done
# you would still have to modify the labels
# and add a colorblind friendly palette

# what if we had two numeric x variables?
# ex: X1st, X2nd

ggplot(data = cdata) + 
  geom_point(aes(x = X1st., y = X2nd., colour = Result), alpha = 0.5) + 
  scale_colour_grey() +
  theme_bw() + 
  labs(x = "First Serve In",
       y = "Second Serves In")

# next up: fit the tree

# ------- data preparation -------

# create training index vector

# need to set seed (reproducability)
# this will ensure we all get the same training/testing datasets
# this is common in industry

RNGkind(sample.kind = "default") # not strictly necessary 
set.seed(2291352) # only matters that we use the same number

train.idx = sample(x = c(1:nrow(cdata)), size = .8*nrow(cdata))

# make training data
train.df = cdata[train.idx,]

# make testing data frame
test.df = cdata[-train.idx,]

# --------- tree fitting / interpretation --------

# set.seed(172172172)
# ctree = rpart(Result ~ Higher.Rank + A. + DF. + Rd +
#                 X1stIn + X1st. + X2nd. + Surface + Rl +
#                 vRk + month + year,
#               data = train.df, # train.df NOT cdata
#               method = "class")

# OR

set.seed(172172172)
ctree = rpart(Result ~ ., # use with caution
              data = train.df, # train.df NOT cdata
              method = "class")

# ctree is our tree object

# plot the tree to visualize
rpart.plot(ctree)

# sometimes trees get too big to visualize
# but we still want to understand
# some of the most important splitting rules

ctree

levels(train.df$Result)

# ----- tuning the tree -------

# R keeps track of all sub trees ie all
# trees smaller than its default
# for each of those sub trees it calculates
# what the prediction error would be, had it
# stopped growing at that particular sub tree

printcp(ctree)

# each row of this output is a sub tree
# row 1 is the simplest - the root node
# the last row is the most complex/biggest tree
# ie the last row is the default tree
# DO look at xerror: this is cross validation error
# we care about xerror because it estimates what the error
# would be OUT OF SAMPLE
# DONT LOOK AT REL ERROR - this estimates IN SAMPLE error,
# which would result in overfitting (youd get a huge tree)
# DO prune your tree to the sub tree that has the 
# smallest xerror (cross validation error, ie out of sample error)

# note: xerror was minimized at the largest tree that R
# decided to fit (and that was based on defualts)
# best practice: grow a BIG BIG TREE BIGGER THAN YOU THINK
# IT EVER SHOULD BE. and then prune it back

# grow a big tree:

set.seed(172172172)
ctree = rpart(Result ~ ., # use with caution
              data = train.df, # train.df NOT cdata
              method = "class",
              control = rpart.control(cp = 0.0001, minsplit = 1))
# under on circumstances are you going to use this to predict
# we need to tune it 

printcp(ctree)

optimalcp = ctree$cptable[which.min(ctree$cptable[,"xerror"]),"CP"]


# now prune back to the tree
# note, we give our prune our OVERGROWN tree
ctree2 = prune(ctree, cp = optimalcp)
rpart.plot(ctree2)

# ------- model validation + prediction ----------

# it is common for people in industry to use the following code
# however for reasons we will discuss, this code is not ideal

# add a column of predictions to our test data set
# going to call new column result_pred
test.df$result_pred = predict(ctree2, test.df, type = "class")

# make confusion matrix
table(test.df$result_pred, test.df$Result)

# note: the code above using the predict function assumes the following:
# if the predicted probability of W > 0.5 it predicts W
# if the predicted probability of W < 0.5 it predicts L

# VARY THE PROBABILITY CUTOFF pi*

# get pi-hats: pick the probabilities of W's
# recall from the notes, W is our positive event
pi_hat = predict(ctree2, test.df, type = "prob")[, "W"]

# how well do we predict with the model on test data?
# if cutoff p = 0.10

p = 0.10
y_hat = as.factor(ifelse(pi_hat > p, "W", "L"))
# make new confusion magrix based on pi* = 0.1
table(y_hat, test.df$Result)

# sensitivity = 96/(96+9) = 0.91428
# specificity = 9/(9+5) = 0.6428571

# if cutoff p = 0.50
p = 0.50
y_hat = as.factor(ifelse(pi_hat > p, "W", "L"))
# make new confusion magrix based on pi* = 0.1
table(y_hat, test.df$Result)

# sensitivity = 0.9048
# specificity = 0.643

# if cutoff p = 0.90
p = 0.90
y_hat = as.factor(ifelse(pi_hat > p, "W", "L"))
# make new confusion magrix based on pi* = 0.1
table(y_hat, test.df$Result)

# sensitivity = 95/(10+95) = 0.9048
# specificity = 12/(12 ÷ 2) = 0.8571

# You will always see a trade-of+ between sens. and spec. as bi* shirts.
# ideally, we Look at sensitivity, specificity for ALL possible
# cutoff values... but this would be awfully tedious using the above method
# -> enter OC curve

# ROC curve
# note: you have to be aware of your 'positive' event at this step
# in the notes, we decided "W" (a win) was our positive event roccurve
rocCurve = roc(response = test.df$Result, # supply truth (in test set)
    predictor = pi_hat, # supply predicted PROBABILITIES of positive case
    levels = c("L", "W") # (negative, positive)
)
# plot basic ROC curve
plot(rocCurve, print.thres = "all", print.auc = TRUE)
# printing threshold results, key: pi^* (spec, sens)
# print.thres = TRUE: threshold with highest sens + spec displayed
# print.thres = "all": all thresholds

# in industry, you might present: 
plot(rocCurve, print.thres = TRUE, print.auc = TRUE)
# simple and informative
# want AUC to be large

# if we set pi^* = 0.861, we can achieve a specificity of 0.857,
# and sensitivity of 0.905

# that is, we will predict a loss 85.7% of the time when Serena
# actually loses.
# we will predict a win 90.5% of the time when Serena actually
# wins
# area under curve is 0.928

# obtain predictions consistent with the above premises:
pi_star = coords(rocCurve, "best", ret = "threshold")$threshold[1]
test.df$result_pred = as.factor(ifelse(pi_hat > pi_star, "W", "L"))

# NOTE: the above is fundamentally different than the following
# test.df$result_pred = predict(ctree2, test.df, type = "class)
# because that code assumes a cutoff of 0.5 (which is not optimal)

# in summary, in industry, accuracy alone is not enough
# and may even be irrelevant. better to present:
# ROC curve
# "best" sensitivity, specificity (print.thres = TRUE)
# AUC (closer to 1 is better)

# given this, your model will either be put into production
# or sent back for further improvement.

# pros and cons of trees: 

# pros:
# easy to interpret
# easily understood by non data scientists
# transparent in how they are fit

# cons: 
# prone to overfitting
# needs to be careful tuning
# instability - my tree can be different from your tree



