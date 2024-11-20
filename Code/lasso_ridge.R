rm(list = ls())

library(tidyverse) # for plotting
library(pROC)
# new packages below
library(glmnet) # for fitting lasso, ridge regressions
library(lubridate) # for easily manipulating dates

f <- read.csv('Data/senators_topics.csv', stringsAsFactors = TRUE)
# prerequisite reading material: forecastinf forgein exchange rates
# problem: this is a super wide data set - there are many x variables
# putting all of these in a MLE logisitc regression model would result
# in unstable predictions
# solution: find a simpler/more parsimonious model

# we already know several ways to find simpler models
# 1: random forest variable importance plot + AIC/BIC
# 2: PCA - reduces the dimension of the x matrix
# 3: backwards/stepwise regression
# 4: clustering - cluster your x variables down to one
#                 manageable cluster column (treated as factor)

# in this code, we will learn a 5th method: lasso and ridge regression

# in summary, lasso/ridge regressions fit GLMS (random + systematic components
# that you already know about) BUT: lasso and ridge regression use a 
# different estimation method - a penalized version that
# doesn't let the beta's get as big as they would otherwise with MLE

# in the paper we practiced stock/exchange rate data using google trends
# so we had a normal random component and the identity link
# here we are predicting popularity (0/1) so we will use 
# a bernoulli and logit link

f <- f %>%
  mutate(
    Date = ymd(created_at),
    month = month(Date) %>% as.factor(),
    popular_bin = ifelse(popular == "Popular", 1, 0)
  )

# remove the columns we don't want to use as predictors or a y variable

f <- f %>% select(-c(popular, created_at))

# this is a prediction problem so we have to split our data into train/test

RNGkind(sample.kind = "default")
set.seed(23591)
train.idx <- sample(x = 1:nrow(f), size = .7*nrow(f))
train.df <- f[train.idx, ]
test.df <- f[-train.idx, ]
# important noteL the above train.df and test.df are data frame structures
# this will come up later - lasso/ridge requires matrix structures

# fit a traditional logistic regression fit with MLE

lr_mle <- glm(popular_bin ~ .,
                data = train.df,
              family = binomial(link = 'logit'))
# the warning message means we have complete seperation
# since we only care about predictions, we are going to roll with it

# look at the coefficients from the MLE logitstic regression
bate <- lr_mle %>% coef()
# if you want to exponentiate
# lr_mle <- %>% coef() %>% exp()

# now time for lasso and ridge regression
# these regression systematically shrink unimportant coefficients to 0
# unimportant - they don't help the model predict out of sample

# note: reacall that RF, trees, logistic regression with MLE, all took
# in training / testing data frames
# that changes with lasso and ridge - they require matrices
# part of that means one-hot-coding any factors

x.train <- model.matrix(popular_bin ~ ., data = train.df)[, -1]
x.test <- model.matrix(popular_bin ~ ., data = test.df)[, -1]

# x.train and x.test have the same info as train.df and test.df, but
# they are matrices

# also need to make vectors of the 0/1 y variable
y.train <- as.vector(train.df$popular_bin)
y.test <- as.vector(test.df$popular_bin)

# use cross validation to fit lasso and ridge regression
lr_lasso_cv <- cv.glmnet(x.train, # train matrix - without y column
                         y.train, # train y vector - y column
                         family = binomial(link = 'logit'), # random + systematic
                         alpha = 1) 

lr_ridge_cv <- cv.glmnet(x.train, # train matrix - without y column
                         y.train, # train y vector - y column
                         family = binomial(link = 'logit'), # random + systematic
                         alpha = 0) 

# these models try a range of lambda values (differing penalty parameters)
# and then use CV to estimate out of sample error at each lambda

plot(lr_lasso_cv)
# choose the lambda value that minimizes out of sample error,
# ie GLM deviance
plot(lr_ridge_cv)

best_lasso_lambda <- lr_lasso_cv$lambda.min
best_ridge_lambda <- lr_ridge_cv$lambda.min

# see the coefficients for the model that minimizes out of sample error
# note: you wouldn't have to do this in RL, but it helps to understand hting

lr_lasso_coef <- coef(lr_lasso_cv, s = "lambda.min") %>% as.matrix()
lr_ridge_coef <- coef(lr_ridge_cv, s = "lambda.min") %>% as.matrix()

# again, understanding the differences between models:

ggplot() + 
  geom_point(aes(lr_ridge_coef, lr_lasso_coef)) + 
  geom_abline(aes(slope = 1, intercept = 0)) + 
  xlim(c(-10, 10)) + 
  ylim(c(-10, 10))

# if time allows compare to MLE coefficients too

# so, while the models (random and systematic) are all the same
# but, the coefficients are vastly different because of the penalization methods

# fit final lasso + ridge models

final_lasso <- glmnet(x.train, y.train, 
                      family = binomial(link = 'logit'),
                      alpha = 1, # for lasso
                      lambda = best_lasso_lambda) # tune model based on cv


final_ridge <- glmnet(x.train, y.train, 
                      family = binomial(link = 'logit'),
                      alpha = 0, # for ridge
                      lambda = best_ridge_lambda) # tune model based on cv

# quantify prediction performance of all 3 models
# note: please call this something other than test.df
# ie, don't overwrite test.df
test.df.preds <- test.df %>%
  mutate(
    mle_pred = predict(lr_mle, test.df, type = 'response'),
    # note: lasso gets the matrix
    lasso_pred = predict(final_lasso, x.test, type = 'response')[,1],
    # note: ridge gets the matrix
    ridge_pred = predict(final_ridge, x.test, type = 'response')[,1]
    # note: all need type = 'response' so we don't use log odds
  )

# fyi: you'll do this (the above) twice on final project for a given y variable
# you'll do it once on CPS testing data set
# you'll do it again on the ACS (real testing set without the truth)


# fit roc curves
mle_rocCurve <- roc(response = as.factor(test.df.preds$popular_bin), # truth
                    predictor = test.df.preds$mle_pred, # predicted preds of MLE
                    levels = c('0', '1')) # positive event comes second

lasso_rocCurve <- roc(response = as.factor(test.df.preds$popular_bin), # truth
                    predictor = test.df.preds$lasso_pred, # predicted preds of MLE
                    levels = c('0', '1')) # positive event comes second

ridge_rocCurve <- roc(response = as.factor(test.df.preds$popular_bin), # truth
                      predictor = test.df.preds$ridge_pred, # predicted preds of MLE
                      levels = c('0', '1')) # positive event comes second

lasso_rocCurve <- roc(response = as.factor(test.df.preds$popular_bin), # truth
                    predictor = test.df.preds$lasso_pred, # predicted preds of MLE
                    levels = c('0', '1')) # positive event comes second


#make data frame of MLE ROC info
mle_data <- data.frame(
  Model = "MLE",
  Specificity = mle_rocCurve$specificities,
  Sensitivity = mle_rocCurve$sensitivities,
  AUC = as.numeric(mle_rocCurve$auc)
)
#make data frame of lasso ROC info
lasso_data <- data.frame(
  Model = "Lasso",
  Specificity = lasso_rocCurve$specificities,
  Sensitivity = lasso_rocCurve$sensitivities,
  AUC = lasso_rocCurve$auc %>% as.numeric
)
#make data frame of ridge ROC info
ridge_data <- data.frame(
  Model = "Ridge",
  Specificity = ridge_rocCurve$specificities,
  Sensitivity = ridge_rocCurve$sensitivities,
  AUC = ridge_rocCurve$auc%>% as.numeric
)

# Combine all the data frames
roc_data <- rbind(mle_data, lasso_data, ridge_data)


# Plot the data
ggplot() +
  geom_line(aes(x = 1 - Specificity, y = Sensitivity, color = Model),data = roc_data) +
  geom_text(data = roc_data %>% group_by(Model) %>% slice(1), 
            aes(x = 0.75, y = c(0.75, 0.65, 0.55), colour = Model,
                label = paste0(Model, " AUC = ", round(AUC, 3)))) +
  scale_colour_brewer(palette = "Paired") +
  labs(x = "1 - Specificity", y = "Sensitivity", color = "Model") +
  theme_minimal()













