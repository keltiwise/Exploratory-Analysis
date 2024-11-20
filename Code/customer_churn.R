rm(list = ls())

library(randomForest)
library(ggplot2)
library(pROC)
library(RColorBrewer)

# ------ 1 --------

# read in customer churn data
cc = read.csv("Data/customer_churn.csv", stringsAsFactors = TRUE)

# goal: predict customer churn (leaving) AND understand what drives a customer to leave
# i.e. give client a predictive model as well as descriptive insights

# see structure of the data
str(cc)
# need to remove Customer ID before we train our model

summary(cc)
# note: there are 9 missing values in Total.Charges
# random forests can't take missing values
# we'll have to 1) remove these values or 2) impute values for them
# either of these options are common in industry
# it's generally best to avoid throwing out data
# so - we'll impute

# imputing 9 missing values of total charges with the median
cc$Total.Charges[is.na(cc$Total.Charges)] = median(cc$Total.Charges, na.rm = TRUE)
summary(cc)

# ---- preparation -----
# set the seed
RNGkind(sample.kind = "default")
set.seed(2291352)

train.idx = sample(x = 1:nrow(cc), size = .7*nrow(cc))
train.df = cc[train.idx, ]
# remove Customer ID
train.df = subset(train.df, select = -c(CustomerID))
# don't have to remove customer ID from testing data
test.df = cc[-train.idx, ]

# fit a baseline forest - not really necessary but helps us estimate
# how long it will take to tune

tempforest = randomForest(Churn ~ ., 
                          data = train.df,
                          ntree = 1000,
                          mtry = 4)

# took a few seconds to run
# because it took a few seconds we might want to consider this when tuning
# it will take a while to test m = 1, 2, 3, 4, ..., 19

dim(train.df)
# what mtry values to consider?
mtry = seq(from = 1, to = 19, by = 3) # to slim down number of tuning options

keeps = data.frame(m = rep(NA, length(mtry)), 
                   OOB_err_rate = rep(NA, length(mtry)))

for (idx in 1:length(mtry)) { 
  print(paste0("Trying m = ", mtry[idx]))
  
  tempforest = randomForest(Churn ~ .,
                            data = train.df,
                            ntree = 1000,
                            mtry = mtry[idx])
  # record iteration's m value in idx'th row
  keeps[idx, "m"] = mtry[idx]
  # record OOB error in idx'th row
  keeps[idx, "OOB_err_rate"] = mean(predict(tempforest) != train.df$Churn)
}

# plot that you can use to justify your chosen tuning parameters

ggplot(data = keeps) + 
  geom_line(aes(x = m, y = OOB_err_rate)) + 
  theme_bw() + labs(x = "m (mtry) value", y = "OOB Error Rate (Minimize)") + 
  scale_x_continuous(breaks = c(1:19))

# 4 us the m that minimizes OOB error rate. this is not a surprise
# since we usually expect somewhere around sqrt(k) to be the best

finalforest = randomForest(Churn ~ .,
                           data = train.df,
                           ntree = 1000,
                           mtry = 4, # chosen from tuning
                           importance = TRUE)

# validate model as a predictive tool

pi_hat = predict(finalforest, test.df, type = "prob")[,"Yes"] # CHOOSE POSITIVE EVENT

rocCurve = roc(response = test.df$Churn,
               predictor = pi_hat,
               levels = c("No", "Yes")) # MAKE SURE THIS LINES UP WITH CHOSEN POSITIVE EVENT

plot(rocCurve, print.thres = TRUE,  print.auc = TRUE)

# what can we say from this?
# AUC = 0.835
# pi* = 0.226 (wow, way off from 0.5, the default threshold)
# we will only predict a customer churns if their probability is greater than 0.226
# sensitivity: we correctly predict a customer churning about 
# 82.19% of the time when they actually do churn
# specificity: we correctly predict a customer remaining loyal about
# 70.2% of the time when they actually stay with the company

# save our predictions in our test set
pi_star = coords(rocCurve, "best", ret = "threshold")$threshold[1]

# create a column of predicted values for test set
test.df$churn_pred = ifelse(pi_hat > pi_star, "Yes", "No")

# note: the above is very much different than 
# test.df$churn_pred = predict(finalforest, test.df, type = "class") <- NOT GOOD
# that code assumes a pi star of 0.5 and does not guarantee the optimal sensitivity and specificity

# that's it for prediction - now for descriptive modeling

varImpPlot(finalforest, type = 1)

vi = as.data.frame(varImpPlot(finalforest, type = 1))
vi$Variable = rownames(vi)

ggplot(data = vi) + 
  geom_bar(aes(x = reorder(Variable, MeanDecreaseAccuracy), weight = MeanDecreaseAccuracy), 
           position = "identity") + 
  coord_flip() + 
  labs(x = "Variable Name", y = "Mean Decrease Accuracy")

# in predicting whether or not a customer churns, tenure and total charges
# and contract are the most important

# but - what way (directionally) do these factors affect the 
# probability that a customer churns

# ---> use logistic regression + plots to help describe this

# goal: fit a parsimonious logistic regression using out of sample
# information to help inform which variables to include

# create Y variable - binary

cc$Churn_bin = ifelse(cc$Churn == "Yes", 1, 0)

m1 = glm(Churn_bin ~ Tenure + Total.Charges,
         data = cc,
         family = binomial(link = "logit"))

BIC(m1) # 6240.222

m2 = glm(Churn_bin ~ Tenure + Total.Charges + 
           Contract + Monthly.Charges,
         data = cc,
         family = binomial(link = "logit"))

BIC(m2) # 5700.782 - smaller is better

m3 = glm(Churn_bin ~ Tenure + Total.Charges + 
           Contract + Monthly.Charges + Internet.Service,
         data = cc,
         family = binomial(link = "logit"))

BIC(m3) # 5636.201 - smaller is better

m4 = glm(Churn_bin ~ Tenure + Total.Charges + 
           Contract + Monthly.Charges + Internet.Service + Tech.Support,
         data = cc,
         family = binomial(link = "logit"))

BIC(m4) # 5614.959 - smaller is better

m5 = glm(Churn_bin ~ Tenure + Total.Charges + 
           Contract + Monthly.Charges + Internet.Service + Tech.Support 
         + Online.Security,
         data = cc,
         family = binomial(link = "logit"))

BIC(m5) # 5583.207 - smaller is better

m6 = glm(Churn_bin ~ Tenure + Total.Charges + 
           Contract + Monthly.Charges + Internet.Service + Tech.Support 
         + Online.Security + Online.Backup,
         data = cc,
         family = binomial(link = "logit"))

BIC(m6) # 5585.866 - bigger - worse!

# m5 is the model with the lowest BIC

summary(m5)

coef(m5)

# interpret what happens every time a customer stays on 
# for another 12 months (1 year)

exp(12 * -0.0584308197)
# 0.4960047

# the odds of churning decrease by about 50% every time 
# a customer has been with the company for another 12 months 

# interpretation statements are more impactful when accompanied with a plot

ggplot(data = cc) + 
  geom_histogram(aes(x = Tenure, fill = Churn), position = "fill") + 
  scale_fill_grey() + labs(x = "Tenure (in months)", y = "Proportion")

# include total.charges (numeric)

ggplot(data = cc) +
  geom_point(aes(x = Tenure, y = Total.Charges/Tenure, colour = Churn),
             alpha = (0.5)) + scale_color_brewer(palette = "Dark2") + 
  labs(x = "Tenure (months", y = "Average Monthly Charges (US $)")









