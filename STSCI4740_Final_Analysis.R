#### Load Libraries ####
library(tidyverse)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(FactoMineR)
library(caret)
library(randomForest)
library(forecast)
library(e1071)
library(pROC)
library(kappaSize)
library(stats)
library(ggplot2)
library(corrplot)
library(FactoMineR)
library(dplyr)
library(tidyverse)
library(caret)
library(stringr)
library(caret)
library(e1071)
library(randomForest)
library(gbm)
library(mlbench)
library(glmnet)
library(pROC)
library(tidyr)
library(irr)
library(MLmetrics)
library(kappaSize)
library(MASS)

#Importing train/test datasets (see python code for split and explanatory analysis)--------

#Note the data is normalized and we also have a dummy variable for wine quality being good (>=6) or poor (<6).

red_wine_train <- read.csv("/Users/chris/Downloads/Data/RW_Train.csv") %>% dplyr::select(-X) %>% rename(good_quality = quality)
red_wine_test <- read.csv("/Users/chris/Downloads/Data/RW_Test.csv") %>% dplyr::select(-X) %>%  rename(good_quality = quality)
white_wine_train <- read.csv("/Users/chris/Downloads/Data/WW_Train.csv") %>% dplyr::select(-X) %>% rename(good_quality = quality)
white_wine_test <- read.csv("/Users/chris/Downloads/Data/WW_Test.csv") %>% dplyr::select(-X) %>% rename(good_quality = quality)
all_wine_train <- read.csv("/Users/chris/Downloads/Data/AW_Train.csv") %>% dplyr::select(-X) %>% rename(good_quality = quality)
all_wine_test <- read.csv("/Users/chris/Downloads/Data/AW_Test.csv") %>% dplyr::select(-X) %>% rename(good_quality = quality)


#Functions--------------------------------------------------------------------------
format_string <- function(vec){
  vec <- sort(vec)
  vec[2] <- gsub("\\.", " ", vec[2], fixed = TRUE)
  my_str <- paste(vec, collapse = ", ")
  return(my_str)

  
}
# Model/Variable selection---------------------------------------------------

column_names <- c("Model","Features Selected","Error/Missclassification Rate","Accuracy")

results_table_red <- matrix(nrow = 0,ncol = 4)
results_table_white <- matrix(nrow = 0,ncol = 4)
colnames(results_table_red) <-column_names
colnames(results_table_white) <-column_names



#Model 1: Logistic Regression
# Variable selection method: Backward selection, Forward Selection (using deviance and AIC)
# Model selection metric: AIC
# stepAIC() function used


#Null and Full models for red wines
null.model_red <- glm(good_quality~1,data = red_wine_train %>% dplyr::select(-type), family = binomial) 
full.model_red <- glm(good_quality~.,data = red_wine_train %>% dplyr::select(-type), family = binomial) 

#Forward selection for read wines 
step.forward_red <- stepAIC(null.model_red,direction = "forward",scope = formula(full.model_red))
step.forward_red.probs <- predict(step.forward_red,red_wine_test, type = "response")
glm.pred_one <- rep(0,dim(red_wine_test)[1])
glm.pred_one[step.forward_red.probs > .5] <- 1
table(glm.pred_one,red_wine_test$good_quality)
#Test error rate 
er_forward_rw<- mean(glm.pred_one != red_wine_test$good_quality)
features_forward_rw <- names(step.forward_red$coefficients)[-1]


#Backward selection for red wines
step.backward_red <- stepAIC(full.model_red,direction = "backward",scope = formula(null.model_red))
step.backward_red.probs <- predict(step.backward_red,red_wine_test, type = "response")
glm.pred_two <- rep(0,dim(red_wine_test)[1])
glm.pred_two[step.backward_red.probs > .5] <- 1
table(glm.pred_two,red_wine_test$good_quality)
#Test error rate 
er_backward_rw <- mean(glm.pred_two != red_wine_test$good_quality)
features_backward_rw <- names(step.backward_red$coefficients)[-1]

#Null and Full models for red wines
null.model_white <- glm(good_quality~1,data = white_wine_train %>% dplyr::select(-type), family = binomial) 
full.model_white <- glm(good_quality~.,data = white_wine_train %>% dplyr::select(-type), family = binomial) 

#Forward selection for white wines
step.forward_white <- stepAIC(null.model_white,direction = "forward",scope = formula(full.model_white))
step.forward_white.probs <- predict(step.forward_white,white_wine_test, type = "response")
glm.pred_three <- rep(0,dim(white_wine_test)[1])
glm.pred_three[step.forward_white.probs > .5] <- 1
table(glm.pred_three,white_wine_test$good_quality)
#Test error rate 
er_forward_ww<- mean(glm.pred_three != white_wine_test$good_quality)
features_forward_ww <- names(step.forward_white$coefficients)[-1]

#Backward selection for white wines
step.backward_white <- stepAIC(full.model_white,direction = "backward",scope = formula(null.model_white))
step.backward_white.probs <- predict(step.backward_white,white_wine_test, type = "response")
glm.pred_four <- rep(0,dim(white_wine_test)[1])
glm.pred_four[step.backward_white.probs > .5] <- 1
table(glm.pred_four,white_wine_test$good_quality)
#Test error rate 
er_backward_ww<- mean(glm.pred_four != white_wine_test$good_quality)
features_backward_ww <- names(step.backward_white$coefficients)[-1]

#Forward selection performs better as compared to backward selection. Forward selection for white wines had the lowest misclassification rate of 0.2584728)

#Model 2: Lasso 
# Lasso performs variable selection, we also have to select tuning parameter through cross validation approach

#Implementing Lasso for red wines 

# Find the best lambda using cross-validation
x <- red_wine_train %>% dplyr::select(-type)
x <- model.matrix(good_quality~.,x)[,-1]
x <-scale(x)
y <- red_wine_train$good_quality

set.seed(10)
fit<- cv.glmnet(x,y,alpha = 1, family = "binomial")
model_rw <- glmnet(x,y, alpha = 1, family = "binomial",lambda = fit$lambda.min)

coef.glmnet(model_rw)
x.test <- red_wine_test %>% dplyr::select(-type)
x.test <- model.matrix(good_quality~.,x.test)[,-1]
x.test <- scale(x.test)
probabilities_one <- predict.glmnet(model_rw, newx = x.test)
predicted.classes_one <- ifelse(probabilities_one > 0.5, 1, 0)
# Model accuracy
observed.classes <- red_wine_test$good_quality
er_lasso_rw<- mean(predicted.classes_one != observed.classes)



#Implementing Lasso for white wines
x <- white_wine_train %>% dplyr::select(-type)
x <- model.matrix(good_quality~.,x)[,-1]
x <-scale(x)
y <- white_wine_train$good_quality

set.seed(10)
fit<- cv.glmnet(x,y,alpha = 1, family = "binomial")
model_ww <- glmnet(x,y, alpha = 1, family = "binomial",
                lambda = fit$lambda.min)

coef.glmnet(model_ww)
x.test <- white_wine_test %>% dplyr::select(-type)
x.test <- model.matrix(good_quality~.,x.test)[,-1]
x.test <- scale(x.test)
probabilities_two <- predict.glmnet(model_ww, newx = x.test)
predicted.classes_two <- ifelse(probabilities_two > 0.5, 1, 0)
# Model accuracy
observed.classes <- white_wine_test$good_quality
er_lasso_ww<- mean(predicted.classes_two != observed.classes)


#Results table 
results_table_red <- results_table_red %>% 
  rbind(list("Logistic Regression,Forward Stepwise Selection",format_string(features_forward_rw),(er_forward_rw),(1-er_forward_rw))) %>% 
  rbind(list("Logistic Regression,Backward Stepwise Selection",format_string(features_backward_rw),(er_backward_rw),(1-er_backward_rw))) %>% 
  rbind(list("Lasso","All features included",(er_lasso_rw),(1-er_lasso_rw))) 

results_table_white <- results_table_white %>% 
  rbind(list("Logistic Regression,Forward Stepwise Selection",format_string(features_forward_ww),(er_forward_ww),(1-er_forward_ww))) %>% 
  rbind(list("Logistic Regression,Backward Stepwise Selection",format_string(features_backward_ww),(er_backward_ww),(1-er_backward_ww))) %>% 
  rbind(list("Lasso","All features included",(er_lasso_ww),(1-er_lasso_ww))) 







