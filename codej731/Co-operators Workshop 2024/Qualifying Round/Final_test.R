library(knitr)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(scales)
library(Rmisc)
library(ggrepel)
library(randomForest)
library(psych)
library(xgboost)
library(glmnet)
library(Matrix)

################### Rest model test
rest_ = read.csv("~/Downloads/Qualification_Package/Claims_Years_1_to_3.csv")
rest_$drv_age2 <- NULL
rest_$drv_age_lic2 <- NULL
rest_$drv_sex2 <- NULL
rest_$vh_make_model <- NULL

get_outliers = function(x){
  which(x > quantile(x)[4] + 1.5*IQR(x) | x < quantile(x)[2] - 1.5*IQR(x))
}

outliers <- get_outliers(rest_$claim_amount)
rest <- rest_[-outliers, ]

testData = read.csv("~/Downloads/Qualification_Package/Submission_Data_modified.csv")
ids <- testData$id_policy
testData$id_policy <- NULL

INDEP_VAR <- c('year', 'pol_no_claims_discount', 'pol_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_age1', 'drv_age_lic1', 'drv_drv2', 'vh_age', 'vh_fuel', 'vh_type', 'vh_speed', 'vh_value', 'vh_weight', 'population', 'town_surface_area')
DEP_VAR <- 'claim_amount'
CAT_VAR <- c('pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_drv2', 'vh_fuel', 'vh_type')
NUM_VAR <- c('year', 'pol_no_claims_discount', 'pol_duration', 'drv_age1', 'drv_age_lic1', 'vh_age', 'vh_speed', 'vh_value', 'vh_weight', 'population', 'town_surface_area')
MODEL_NAME <- 'gamma-glm.joblib'

for (i in 1:nrow(testData)){
  if(is.na(testData$vh_speed[i])){
    testData$vh_speed[i] <- as.integer(median(testData$vh_speed, na.rm=TRUE))
  }
  if(is.na(testData$vh_value[i])){
    testData$vh_value[i] <- as.integer(median(testData$vh_value, na.rm=TRUE))
  }
  if(is.na(testData$vh_weight[i])){
    testData$vh_weight[i] <- as.integer(median(testData$vh_weight, na.rm=TRUE))
  }
  if(testData$drv_drv2[i] == "No"){
    testData$drv_age2[i] <- 0
    testData$drv_age_lic2[i] <- 0
    testData$drv_sex2[i] <- 1
  }
}

for (i in 1:nrow(rest)){
  if(is.na(rest$vh_speed[i])){
    rest$vh_speed[i] <- as.integer(median(rest$vh_speed, na.rm=TRUE))
  }
  if(is.na(rest$vh_value[i])){
    rest$vh_value[i] <- as.integer(median(rest$vh_value, na.rm=TRUE))
  }
  if(is.na(rest$vh_weight[i])){
    rest$vh_weight[i] <- as.integer(median(rest$vh_weight, na.rm=TRUE))
  }
  if(rest$drv_drv2[i] == "No"){
    rest$drv_age2[i] <- 0
    rest$drv_age_lic2[i] <- 0
    rest$drv_sex2[i] <- 1
  }
}

# Data preprocessing
X_rest <- model.matrix(~ . - 1, data = rest[, INDEP_VAR]) # Creating design matrix with dummy variables
y_rest <- rest[, DEP_VAR]

X_testData <- model.matrix(~ . - 1, data = testData[, INDEP_VAR])
# Scaling numerical variables
scaled_vars <- scale(X_rest[, NUM_VAR])
scaled_test_vars <- scale(testData[, NUM_VAR])
colnames(scaled_test_vars) <- NUM_VAR
colnames(scaled_vars) <- NUM_VAR

# Combine scaled numerical variables with categorical variables
X_rest <- cbind(X_rest[, !colnames(X_rest) %in% NUM_VAR], scaled_vars)
X_testData <- cbind(X_testData[, !colnames(X_testData) %in% NUM_VAR], scaled_test_vars)

# Training and testing sets
set.seed(123) # For reproducibility
#library(caret)
train_index <- createDataPartition(y_rest, p = 0.8, list = FALSE)
X_rest_train <- X_rest[train_index, ]
y_rest_train <- y_rest[train_index]
X_rest_test <- X_rest[-train_index, ]
y_rest_test <- y_rest[-train_index]
default_param<-list(
  objective = "reg:gamma",
  booster = "gblinear",
  eta=0.1, #default = 0.3
  gamma=0.05,
  max_depth=3, #default=6
  min_child_weight=3, #default=1
  subsample=1,
  colsample_bytree=1
)

dtrain <- xgb.DMatrix(data = as.matrix(X_rest_train), label= y_rest_train)
dtest <- xgb.DMatrix(data = as.matrix(X_rest_test))
dtestSubmissions <- xgb.DMatrix(data = as.matrix(X_testData))

xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 5000, verbose=TRUE, maximize = FALSE)
pred_rest_xgb <- predict(xgb_mod, dtest)
pred_sub_xgb <- predict(xgb_mod, dtestSubmissions)

# Gamma GLM
#library(glmnet)
glm_rest <- glm(y_rest_train ~ ., data=data.frame(X_rest_train), family = "gaussian")

# Random Forest
rf_rest <- randomForest(y_rest_train ~ ., data = X_rest_train,
                        ntree=500,
                        mtry = 6,
                        importance = TRUE,
                        proximity = TRUE)

# Predictions
pred_sub_glm <- predict(glm_rest, data.frame(X_testData))
pred_rest_glm <- predict(glm_rest, data.frame(X_rest_test))
pred_sub_rf <- predict(rf_rest, X_testData)
pred_rest_rf <- predict(rf_rest, X_rest_test)

# Lasso and Ridge Regression
set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.5,by = 0.0005))
ridgeGrid <- expand.grid(alpha = 0, lambda = seq(0.001,0.5,by = 0.0005))

lasso_mod <- train(x=X_rest_train, y=y_rest_train, method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
ridge_mod <- train(x=X_rest_train, y=y_rest_train, method='glmnet', trControl= my_control, tuneGrid=ridgeGrid) 

pred_rest_lasso <- predict(lasso_mod,X_rest_test)
pred_sub_lasso <- predict(lasso_mod,X_testData)

pred_sub_ridge <- predict(ridge_mod,X_testData)
pred_rest_ridge <- predict(ridge_mod,X_rest_test)

sub_avg_q1 <- data.frame(claim_amount = (pred_rest_glm+pred_rest_xgb+pred_rest_lasso+pred_rest_ridge+pred_rest_rf)/5)
sub_avg <- data.frame(claim_amount = (pred_sub_glm+pred_sub_xgb+pred_sub_lasso+pred_sub_ridge+pred_sub_rf)/5)

# RMSE calculation
RMSE_rest_glm <- sqrt(mean((y_rest_test - pred_rest_glm)^2))
RMSE_rest_rf <- sqrt(mean((y_rest_test - pred_rest_rf)^2))
RMSE_rest_xgb <- sqrt(mean((y_rest_test - pred_rest_xgb)^2))
RMSE_rest_lasso <- sqrt(mean((y_rest_test - pred_rest_lasso)^2))
RMSE_rest_ridge <- sqrt(mean((y_rest_test - pred_rest_ridge)^2))
RMSE_rest_avg <- sqrt(mean((y_rest_test - sub_avg_q1$claim_amount)^2))

cat("GLM RMSE:", round(RMSE_rest_glm, 3), "\n")
cat("RF RMSE:", round(RMSE_rest_rf, 3), "\n")
cat("XGB RMSE:", round(RMSE_rest_xgb, 3), "\n")
cat("lasso RMSE:", round(RMSE_rest_lasso, 3), "\n")
cat("ridge RMSE:", round(RMSE_rest_ridge, 3), "\n")
cat("avg RMSE:", round(RMSE_rest_avg, 3), "\n")

cat("Pred Mean:", round(mean(sub_avg$claim_amount), 3), "\n")
cat("Actual Mean:", round(mean(y_rest_test), 3), "\n")
cat("GLM RMSE / Actual Mean", round(RMSE_rest_glm / mean(y_rest_test), 3), "\n")
cat("RF RMSE / Actual Mean", round(RMSE_rest_rf / mean(y_rest_test), 3), "\n")
cat("XGB RMSE / Actual Mean", round(RMSE_rest_xgb / mean(y_rest_test), 3), "\n")
cat("lasso RMSE / Actual Mean", round(RMSE_rest_lasso / mean(y_rest_test), 3), "\n")
cat("ridge RMSE / Actual Mean", round(RMSE_rest_ridge / mean(y_rest_test), 3), "\n")
cat("avg RMSE / Actual Mean", round(RMSE_rest_avg / mean(y_rest_test), 3), "\n")

compare_predictions = function(var,data){
  
  agg = data %>% group_by(!!sym(var)) %>% summarise(Severity = mean(claim_amount),Prediction = mean(l1))
  
  ggplot(agg) +  aes_string(var,"Severity",group = 1) + geom_point() + geom_line(aes(color = "Actual")) +  
    geom_point(aes_string(var,"Prediction",group = 1)) + geom_line(aes_string(var,"Prediction"))
  
}

l1 <- data.frame(drv_age1=rest$drv_age1[-train_index], claim_amount=y_rest_test, l1 = pred_rest_xgb)
compare_predictions("drv_age1", l1)

final_results <- data.frame(id_policy = ids, claim_amount = pred_sub_xgb)
write.csv(final_results, "~/Downloads/Qualification_Package/submission_xgb_v3.csv")
