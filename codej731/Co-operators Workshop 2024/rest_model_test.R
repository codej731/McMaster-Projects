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
#rest_$drv_age2 <- NULL
#rest_$drv_age_lic2 <- NULL
#rest_$drv_sex2 <- NULL
#rest_$vh_make_model <- NULL

get_outliers = function(x){
  which(x > quantile(x)[4] + 1.5*IQR(x) | x < quantile(x)[2] - 1.5*IQR(x))
}

outliers <- get_outliers(rest_$claim_amount)

drop <- c("drv_age2", "drv_age1", "drv_age_lic1", "drv_age_lic2", "vh_make_model", "town_surface_area", "population")
#drop columns
rest <- rest_[, !(names(rest_) %in% drop)]

rest$drv_age_combined <- rest_$drv_age2 + rest_$drv_age1 - rest_$drv_age_lic1 - rest_$drv_age_lic2
rest$popl_per_sa <- rest_$population / rest_$town_surface_area

testData_ = read.csv("~/Downloads/Qualification_Package/Dislocation_dataset.csv", stringsAsFactors = F)
dropTest <- c("drv_age2", "drv_age1", "drv_age_lic1", "drv_age_lic2", "vh_make_model", "town_surface_area", "population", "Frequency", "Old_Severity_Estimate", "Fixed_UW_Expense", "Variable_Expense", "Target_UW_Profit")

ids <- testData_$id_policy
testData_$id_policy <- NULL

testData <- testData_[, !names(testData_) %in% drop]
testData$X <- NULL

testData$drv_age_combined <- testData_$drv_age2 + testData_$drv_age1 - testData_$drv_age_lic1 - testData_$drv_age_lic2
testData$popl_per_sa <- testData_$population / testData_$town_surface_area

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
    testData$drv_age_combined[i] <- as.integer(median(testData$drv_age_combined, na.rm=TRUE))
    #testData$drv_age_lic2[i] <- 0
    testData$drv_sex2[i] <- "Null"
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
    #rest$drv_age2[i] <- 0
    #rest$drv_age_lic2[i] <- 0
    rest$drv_age_combined[i] <- as.integer(median(rest$drv_age_combined, na.rm=TRUE))
    rest$drv_sex2[i] <- "Null"
  }
}

INDEP_VAR <- c('drv_sex2', 'drv_age_combined', 'year', 'pol_no_claims_discount', 'pol_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_drv2', 'vh_age', 'vh_fuel', 'vh_type', 'vh_speed', 'vh_value', 'vh_weight', 'popl_per_sa')
DEP_VAR <- 'claim_amount'
CAT_VAR <- c('pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_drv2', 'vh_fuel', 'vh_type', 'drv_sex2', 'year')
NUM_VAR <- c('pol_no_claims_discount', 'pol_duration', 'drv_age_combined', 'vh_age', 'vh_speed', 'vh_value', 'vh_weight', 'popl_per_sa')

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

xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 10000, verbose=TRUE, maximize = FALSE)
pred_rest_xgb <- predict(xgb_mod, dtest)
pred_sub_xgb <- predict(xgb_mod, dtestSubmissions)

RMSE_rest_xgb <- sqrt(mean((y_rest_test - pred_rest_xgb)^2))

cat("XGB RMSE:", round(RMSE_rest_xgb, 3), "\n")

cat("Pred Mean:", round(mean(pred_rest_xgb), 3), "\n")
cat("Actual Mean:", round(mean(y_rest_test), 3), "\n")

cat("XGB RMSE / Actual Mean", round(RMSE_rest_xgb / mean(y_rest_test), 3), "\n")

compare_predictions = function(var,data){
  
  agg = data %>% group_by(!!sym(var)) %>% summarise(Severity = mean(claim_amount),Prediction = mean(l1))
  
  ggplot(agg) +  aes_string(var,"Severity",group = 1) + geom_point() + geom_line(aes(color = "Actual")) +  
    geom_point(aes_string(var,"Prediction",group = 1)) + geom_line(aes_string(var,"Prediction"))
  
}
l1 <- data.frame(drv_age_combined=rest$drv_age_combined[-train_index], claim_amount=y_rest_test, l1 = pred_rest_xgb)
compare_predictions("drv_age_combined", l1)

final_results <- data.frame(id_policy = ids, claim_amount = pred_sub_xgb)
write.csv(final_results, "~/Downloads/Qualification_Package/submission_xgb_vvv.csv")
mean(pred_sub_xgb)
