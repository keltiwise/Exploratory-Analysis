rm(list = ls())
library(rpart)
library(rpart.plot)
library(ggplot2)
library(pROC)
library(RColorBrewer)

# ------ 1 -------

# one way to read in data
# specify fill file path into read.csv
f = read.csv("Data/fraud_data.csv", stringsAsFactors = TRUE)

summary(f)

str(f)

# doesn't appear to be any issues with reading in the data
# let's do exploratory analysis to take a closer look

# start with months as customer (numeric) - make histogram

ggplot(data = f) + 
  geom_histogram(aes(x = months_as_customer)) + 
  labs(x = "Months as Customer", y = "Count")

# let's do age of customer (numeric) - make histogram

ggplot(data = f) + 
  geom_histogram(aes(x = age), binwidth = 1) + 
  labs(x = "Age of Customer", y = "Count")

ggplot(data = f) + 
  geom_density(aes(x = age), fill = "cornflowerblue") + 
  labs(x = "Age of Customer", y = "Density")

# let's do incident type (factor) - make barchart

ggplot(data = f) + 
  geom_bar(aes(x = incident_type)) + 
  coord_flip() + 
  labs(x = "Type of Incident", y = "Count")
# we used coor_flip to flip the axes 
# we did this because the x labels are long and were smashed together
# do not do this if you do not need to, only when they are smashed together

# now our Y variable, fraud_reported (factor) - bar chart

ggplot(data = f) + 
  geom_bar(aes(x = fraud_reported)) + 
  labs(x = "Was Fruad Reported?", y = "Count")

# ----- 2 -------

# prepare your data and fit a tuned classification tree
# use seed 1981536 for splitting data
# use seed 172172172 for fitting tree

RNGkind(sample.kind = "default")
set.seed(1981536)
train.idx = sample(x = 1:nrow(f), size = floor(.8*nrow(f)))

# make training data
train.df = f[train.idx,]

# make testing data frame
test.df = f[-train.idx,]

set.seed(172172172)
ctree = rpart(fraud_reported ~ ., # use with caution
              data = train.df, # train.df NOT cdata
              method = "class")

# ----- 3 -------

rpart.plot(ctree)

# problem in our life: there are so many visualizations we could make
# how to choose?
# this tree suggests that incident severity and education level 
# are some of the most important for predicting fraud
# let's visualize

# recall that black and white are always colorblind friendly
# futhur anything monochromatic is always colorblind friendly
# alternatively: RColorBrewer provides many colorblind friendly palettes
# to choose from:

display.brewer.all(colorblindFriendly = TRUE)

ggplot(data = f) + 
  geom_bar(aes(x = incident_severity, fill = fraud_reported), 
           position = "fill") + 
  coord_flip() +
  scale_fill_brewer("Fraud\nReported", palette = "Paired") +
  labs(x = "Severity Level", y = "Proportion")


ggplot(data = f) + 
  geom_bar(aes(x = incident_severity, fill = fraud_reported), 
           position = "fill") + 
  coord_flip() +
  scale_fill_brewer("Fraud\nReported", palette = "Paired") +
  labs(x = "Severity Level", y = "Proportion") + 
  facet_wrap(~insured_education_level)

# we can do better ... there is an inherent ordering to the education level, but this plot 
# doesn't recognize that
# let's reorder that factor

# overwrite the column:
f$insured_education_level = factor(f$insured_education_level, 
                             levels = c("High School", "Associate",
                                        "College", "Masters",
                                        "JD", "MD", "PhD"))

ggplot(data = f) + 
  geom_bar(aes(x = incident_severity, fill = fraud_reported), 
           position = "fill") + 
  coord_flip() +
  scale_fill_brewer("Fraud\nReported", palette = "Paired") +
  labs(x = "Severity Level", y = "Proportion") + 
  facet_wrap(~insured_education_level)

# ----- 4 -------

optimalcp = ctree$cptable[which.min(ctree$cptable[,"xerror"]),"CP"]
ctree2 = prune(ctree, cp = optimalcp)

# ------ 5 -------
rpart.plot(ctree2)

test.df$result_pred = predict(ctree2, test.df, type = "class")

# make confusion matrix
table(test.df$result_pred, test.df$fraud_reported)

pi_hat = predict(ctree2, test.df, type = "prob")[, "Y"]

# ---- 6 -------

rocCurve = roc(response = test.df$fraud_reported, # supply truth (in test set)
               predictor = pi_hat, # supply predicted PROBABILITIES of positive case
               levels = c("N", "Y") # (negative, positive)
)

plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

# ---- 7 -------





