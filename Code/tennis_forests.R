rm(list = ls())

# just once
# install.packages("randomForest")

# each time you open R
library(randomForest)
library(ggplot2)
library(pROC)

# read in data
cdata = read.csv("Data/tennis.csv", stringsAsFactors = TRUE)

# ----- data preperation -------

# set seed
RNGkind(sample.kind = "default")
set.seed(2291352)
# train.idx will contain a random sample of row indices
train.idx = sample(x = 1:nrow(cdata), size = floor(.8*nrow(cdata)))
# make training data
train.df = cdata[train.idx, ]
# the rest will be for testing
test.df = cdata[-train.idx, ]

# ------- baseline forest ------

myforest = randomForest(Result ~ ., # recall notes on this syntax
                        data = train.df, # TRAINING DATA
                        ntree = 1000, # fit B = 1000 seperate classification trees
                        mtry = 4, # choose m - sqrt(12) = 3.464102 - rounded to 4
                        importance = TRUE) # importance can help us identify important 
                                           # predictors (later)

myforest

# note OOB estimate of error rate:
# OOB = out of bad - recall bagging. approximately 1/3 of obs are OOB for each tree

(29 + 401)/(29 + 35 + 401 + 10) # accuracy
# = 0.9052632 
1 - 0.9052632 # OOB
# = 0.0947368 = OOB = 9.47

# this is a fine baseline forest, but it is best to do some tuning
# in the notes, i said more trees is always better. this is true,
# in practice, you don't need to tune ntree. do as many as you
# have time for - usually at least 500 it's a common misconception
# that more trees overfit. forest don't overfit because of too

# a sequence of B (# of trees) that we want to try
# we'll loop through each unique element in the vector below
Btry = c(15, 25, seq(from = 50, to = 100, by = 50))

# make room for B, OOB error
keeps = data.frame(ntree = rep(NA, length(Btry)),
                   OOB_error_rate = rep(NA, length(Btry)))

for (idx in 1:length(Btry)) {
  tempforest = randomForest(Result ~ .,
                            data = train.df,
                            ntree = Btry[idx],
                            mtry = 6)
# record how many trees we trued
keeps[idx, "ntree"] = Btry[idx]
# record what our OOB error rate was
keeps[idx, "OOB_error_rate"] = mean(predict(tempforest) != train.df$Result)

}

qplot(ntree, OOB_error_rate, geom = c("point"), data = keeps) + 
  theme_bw() + labs(x = "Number of Trees", y = "OOB Error Rate")


# a sequence of m (# of explanatories sampled at each tree) that we want to 
# try 
# we'll loop through each unique element in the vector below
# range of m values that we want to try
mtry = c(1:12) # think: what is even possible here? 100? no

# make room for B, OOB error
keeps2 = data.frame(m = rep(NA, length(mtry)),
                   OOB_error_rate = rep(NA, length(mtry)))

for (idx in 1:length(mtry)) {
  tempforest = randomForest(Result ~ .,
                            data = train.df,
                            ntree = 1000,
                            mtry = mtry[idx])
  
  keeps2[idx, "m"] = mtry[idx]
  
  keeps2[idx, "OOB_error_rate"] = mean(predict(tempforest) != train.df$Result)
}

keeps2

ggplot(data = keeps2) + 
  geom_line(aes(m, OOB_error_rate)) + 
  theme_bw() + labs(x = "m (mtry) value", y = "OOB Error Rate")

# you'll typically see a U shape in the plot
# choose the m value with the lowest OOB error rate
# ie the value at the lowest point of U
# for me this was between 4 and 6 as we thought it might be
# recall the theory tells us sqrt(k) tends to be the best

# see notes for time-saving tips when k is large

# final forest based on tuning:
final_forest = randomForest(Result ~ .,
                            data = train.df,
                            ntree = 1000,
                            mtry = 3,
                            importance = TRUE)

# use this for prediction

# ----- results -----

# in most problems our goal emphasizes prediction or interpretation
# goal is prediction: 

# create ROC curve
# assuming positive event is W
library(rpart.plot)
pi_hat = predict(final_forest, test.df, type = "prob")[,"W"]
rocCurve = roc(response = test.df$Result,
               predictor = pi_hat,
               levels = c("L", "W"))

plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# if we set pi* to 0.793
# we can achieve a specificity of 0.929 and sensitivity of 0.838
# more meaningful:
# we will predict a loss 93% of the time when serena actually loses
# we predict a win 84% of the time when serena actually wins
# area under the curve is 0.948

# compare this to the ROC and AUC for our final tree
# we would prefer the forest as AUC is higher

# make a column of predicted values in our test data
pi_star = coords(rocCurve, "best", ret = "threshold")$threshold[1]
pi_star
# overwrite those old predictions with something better
test.df$forest_pred = as.factor(ifelse(pi_hat > pi_star, "W", "L"))

# if our goal is purely prediction, we are done now
# we would report our optimal sensitivity specificity and accuracy
# after setting the optimal threshold of pi* 
# in our proposal to implement our model into production

# ---- interpretation ------
# if however we want to interpret or understand relationships between X and Y
# we can look at relative importance of each of our variables in predicting wins
# note: we can only do this if we specify importance = TRUE in randomForest()
varImpPlot(final_forest, type = 1)
# it looks like are most important
# that is, these are the variables for which prediction would suffer the most if they
# were removed
# sometimes, thought, we want to assess directional effects as well as importance
# this is where a random forest is lacking

# RECALLL OUR DISCUSSIONS ON VARIABLE SELECTIONS IN GLMS

#   Random forest pro: automatic variable selection
#   Random forest con: no directional effects for interpretation
#   Logistic regression pro: we get directional effects
#   Logistic regression con: no automatic variable selection
#   ---> reasons many data scientists don't care for p-value based methods

# pairing random forests and logistic regression is a natrual thing to do to accomplish
# both prediction and understanding
# let's fit a logistic regression using only the most important variables 
# as explanatory variables

# first create a bernoulli RV
cdata$Result_bin = ifelse(cdata$Result == "W", 1, 0)

# fit a bernoulli logistic regression in R
# note: we have paired down the number of x variables using the RF
# thus, creating a data driven parsimonious GLM

m1 = glm(Result_bin ~ X2nd. + X1st. ,
         data = cdata, 
         family = binomial(link = "logit"))

BIC(m1) # 302.11

m2 = glm(Result_bin ~ X2nd. + X1st. + vRk,
         data = cdata, 
         family = binomial(link = "logit"))

BIC(m2)# 295.7352 BIC decreased 

m3 = glm(Result_bin ~ X2nd. + X1st. + vRk + Rd,
         data = cdata, 
         family = binomial(link = "logit"))

BIC(m3) # 331.6165

# we would prefer m2, the model with just X2nd. X1st. and vRk as x variables

# do a summary(m) to check for complete seperation

summary(m2)

# no standard erros > 5, no signs of complete seperation
# collect beta vector
beta = coef(m2)
beta

# ex: interpret opponent rank coefficient:
# holding all other game characteristics constant,
# the odds of serena winning the match increases
# by a factor of exp(0.0195 * 5) = 1.10 for every
# 5-place increase in the rank of the opponent

# that is, her odds increase by about 10%

# exponentiate to get odds rations
exp(beta)
