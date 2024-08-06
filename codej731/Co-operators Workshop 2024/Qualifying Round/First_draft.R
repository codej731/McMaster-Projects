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

RMSE = function(x,y){
  MSE = sum((y - x)^2)/length(x)
  return(MSE^.5)
}

#### Load Claims dataset
claims = read.csv("D:/OneDrive-Mcmaster/OneDrive - McMaster University/桌面/Qualification_Package/Claims_Years_1_to_3.csv", stringsAsFactors = F)
ids <- claims %>% select(id_policy)
claims_amts <- claims %>% select(claim_amount)

claims <- claims %>% select(-id_policy, -vh_make_model, -year)

claims$pol_pay_freq <- as.factor(claims$pol_pay_freq)
claims$pol_payd <- as.factor(claims$pol_payd)
#claims$year <- as.factor(claims$year)
claims$pol_usage <- as.factor(claims$pol_usage)
claims$drv_sex1 <- as.factor(claims$drv_sex1)
claims$vh_fuel <- as.factor(claims$vh_fuel)
claims$vh_type <- as.factor(claims$vh_type)
claims$drv_drv2 <- as.factor(claims$drv_drv2)
claims$drv_sex1 <- as.factor(ifelse(claims$drv_sex1 =="M",2,3))
claims$drv_sex2 <- ifelse(claims$drv_sex2 =="M",2,3)


for (i in 1:nrow(claims)){
  if(is.na(claims$vh_speed[i])){
    claims$vh_speed[i] <- as.integer(median(claims$vh_speed, na.rm=TRUE))
  }
  if(is.na(claims$vh_value[i])){
    claims$vh_value[i] <- as.integer(median(claims$vh_value, na.rm=TRUE))
  }
  if(is.na(claims$vh_weight[i])){
    claims$vh_weight[i] <- as.integer(median(claims$vh_weight, na.rm=TRUE))
  }
  if(claims$drv_drv2[i] == "No"){
    claims$drv_age2[i] <- 0
    claims$drv_age_lic2[i] <- 0
    claims$drv_sex2[i] <- 1
  }
}

claims$drv_sex2 <- as.factor(claims$drv_sex2)

numericVars <- which(sapply(claims, is.numeric)) #index vector numeric variables
numericVarNames <- names(numericVars) #saving names vector for use later on
factorVars <- which(sapply(claims, is.factor)) #index vector factor variables
cat('There are', length(numericVars), 'numeric variables, and', length(factorVars), 'categoric variables')

all_numVar <- claims[, numericVars]
cor_numVar <- cor(all_numVar, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with claim amount
cor_sorted <- as.matrix(sort(cor_numVar[,'claim_amount'], decreasing = TRUE))
#select all corrs
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.001)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]
corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt", tl.cex = 0.7,cl.cex = .7, number.cex=.7)

set.seed(2018)
quick_RF <- randomForest(x=claims[1:5000,-21], y=claims$claim_amount[1:5000], ntree=100,importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:21,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")

numericVarNames <- numericVarNames[!(numericVarNames %in% c('claim_amount'))] #numericVarNames was created before having done anything

DFnumeric <- claims[, names(claims) %in% numericVarNames]

DFfactors <- claims[, !(names(claims) %in% numericVarNames)]
DFfactors <- DFfactors[, names(DFfactors) != 'claim_amount']

cat('There are', length(DFnumeric), 'numeric variables, and', length(DFfactors), 'factor variables')

#for(i in 1:ncol(DFnumeric)){
#  if (abs(skew(DFnumeric[,i]))>0.9){
#    DFnumeric[,i] <- log(DFnumeric[,i] +1)
#  }
#}

PreNum <- preProcess(DFnumeric, method=c("center", "scale"))
print(PreNum)
DFnorm <- predict(PreNum, DFnumeric)
dim(DFnorm)
DFdummies <- as.data.frame(model.matrix(~.-1, DFfactors))
dim(DFdummies)

combined <- cbind(DFnorm, DFdummies)
qqnorm(claims$claim_amount)
qqline(claims$claim_amount)
skew(claims$claim_amount)
claims$claim_amount <- claims$claim_amount
#qqnorm(claims$claim_amount)
#qqline(claims$claim_amount)
#skew(claims$claim_amount)

set.seed(355)
trainIndex <- createDataPartition(claims$claim_amount, p = 0.8, list = FALSE)
train1 <- combined[trainIndex, ]
test1 <- combined[-trainIndex, ]

#############random forest

#set.seed(123)
#claims$claim_amount <- as.factor(claims$claim_amount)
#ind <- sample(2, nrow(claims), replace = TRUE, prob = c(0.7, 0.3))
#train2 <- train1[complete.cases(train1),]
#train2 <- claims[ind == 1,]
#test2 <- claims[ind ==2,]

#df1 <- df %>%
#  mutate(across(c(claims$vh_value,claims$drv_age1,claims$vh_speed,claims$vh_weight,claims$drv_age_lic1,claims$population,claims$pol_no_claims_discount, claims$town_surface_area, claims$pol_duration, claims$drv_age_lic2, claims$vh_age), as.numeric))

#x <- c(claims$vh_value,claims$drv_age1,claims$vh_speed,claims$vh_weight,claims$drv_age_lic1,claims$population,claims$pol_no_claims_discount, claims$town_surface_area, claims$pol_duration, claims$drv_age_lic2, claims$vh_age)
#df[x] <- lapply(df[x], \(i) as.numeric(i))

#train1.imputed <- rfImpute(claims$claim_amount ~., data = claims, iter=6)    
#replace the NA value (no need here if red warning)
set.seed(222)
#rf <- randomForest(claims$claim_amount[trainIndex] ~ vh_value + drv_age1 + vh_speed + 
#                     vh_weight + drv_age_lic1 + population+ town_surface_area+ 
#                     pol_duration+ drv_age_lic2 + vh_age+
#                     pol_no_claims_discount + drv_age2, data = train1,
#                   ntree=500,
#                   mtry = 8,
#                   improtance = TRUE,
#                   proximity = TRUE)
#rf <- randomForest(claims$claim_amount ~ claims$vh_value+claims$drv_age1+claims$vh_speed+claims$vh_weight+claims$drv_age_lic1+claims$population+claims$town_surface_area+claims$pol_duration+ claims$drv_age_lic2+ claims$vh_age, data = train1)
rf <- randomForest(claims$claim_amount[trainIndex] ~ ., data = train1,
                   ntree=500,
                   mtry = 8,
                   importance = TRUE,
                   proximity = TRUE)

rf
#rf$err.rate[,1]
print(rf)
attributes(rf)


# Prediction & Confusion Matrix - train data
p1 <- predict(rf, train1)
head(p1)
head(train1$claim_amount) #show accuracy

#confusionMatrix(p1, train1$claim_amount[trainIndex])

# Prediction & Confusion Matrix - test data
p2 <- predict(rf, test1)
#confusionMatrix(p2, test1$claim_amount)

# Error rate of Random Forest
plot(rf)

# Tune mtry
t <- tuneRF(train1[,-27], train1[,27],
       stepFactor = 0.5,
       plot = TRUE,
       ntreeTry = 300,
       trace = TRUE,
       improve = 0.02)

# No. of nodes for the trees
hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green")

# Find variable importance
varImpPlot(rf)

RMSE = function(x,y){
  MSE = sum((y - x)^2)/length(x)
  return(MSE^.5)
}
######################Lasso regression
set.seed(27042018)
my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))
ridgeGrid <- expand.grid(alpha = 0, lambda = seq(0.001,0.1,by = 0.0005))

lasso_mod <- train(x=train1, y=claims$claim_amount[trainIndex], method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 
ridge_mod <- train(x=train1, y=claims$claim_amount[trainIndex], method='glmnet', trControl= my_control, tuneGrid=ridgeGrid) 
lasso_mod$bestTune
ridge_mod$bestTune

lassoVarImp <- varImp(lasso_mod,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

LassoPred <- predict(lasso_mod, test1)
RidgePred <- predict(ridge_mod, test1)
predictions_lasso <- LassoPred #need to reverse the log to the real values
head(predictions_lasso)
predictions_ridge <- RidgePred

xgb_grid = expand.grid(
  nrounds = 25,
  eta = c(0.1, 0.05, 0.01),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree=1,
  min_child_weight=c(1, 2, 3, 4 ,5),
  subsample=1
)

#xgb_caret <- train(x=train1, y=claims$claim_amount[trainIndex], method='xgbTree', trControl= my_control, tuneGrid=xgb_grid) 
#xgb_caret$bestTune

label_train <- claims$claim_amount[trainIndex]
test_labels <- ids[-trainIndex]

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = as.matrix(train1), label= label_train)
dtest <- xgb.DMatrix(data = as.matrix(test1))

#use best values from caret
default_param<-list(
  objective = "reg:squarederror",
  booster = "gbtree",
  eta=0.05, #default = 0.3
  gamma=0,
  max_depth=3, #default=6
  min_child_weight=3, #default=1
  subsample=1,
  colsample_bytree=1
)

xgbcv <- xgb.cv( params = default_param, data = dtrain, nrounds = 750, nfold = 5, showsd = T, stratified = T, print_every_n = 40, early_stopping_rounds = 10, maximize = F)
xgb_mod <- xgb.train(data = dtrain, params=default_param, nrounds = 454)

XGBpred <- predict(xgb_mod, dtest)
predictions_XGB <- XGBpred #need to reverse the log to the real values
head(predictions_XGB)

library(Ckmeans.1d.dp) #required for ggplot clustering
mat <- xgb.importance (feature_names = colnames(train1), model = xgb_mod)
xgb.ggplot.importance(importance_matrix = mat[1:25], rel_to_first = TRUE)

################### Rest model test
rest = read.csv("D:/OneDrive-Mcmaster/OneDrive - McMaster University/桌面/Qualification_Package/rest.csv")

DATASET_FILE_NAME <- 'training-dataset.csv'
INDEP_VAR <- c('year', 'pol_no_claims_discount', 'pol_duration', 'pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_age1', 'drv_age_lic1', 'drv_drv2', 'vh_age', 'vh_fuel', 'vh_type', 'vh_speed', 'vh_value', 'vh_weight', 'population', 'town_surface_area')
DEP_VAR <- 'claim_amount'
CAT_VAR <- c('pol_pay_freq', 'pol_payd', 'pol_usage', 'drv_sex1', 'drv_drv2', 'vh_fuel', 'vh_type')
NUM_VAR <- c('year', 'pol_no_claims_discount', 'pol_duration', 'drv_age1', 'drv_age_lic1', 'vh_age', 'vh_speed', 'vh_value', 'vh_weight', 'population', 'town_surface_area')
MODEL_NAME <- 'gamma-glm.joblib'

# Data preprocessing
X_rest <- model.matrix(~ . - 1, data = rest[, INDEP_VAR]) # Creating design matrix with dummy variables
y_rest <- rest[, DEP_VAR]

# Scaling numerical variables
scaled_vars <- scale(rest[, NUM_VAR])
colnames(scaled_vars) <- NUM_VAR

# Combine scaled numerical variables with categorical variables
X_rest <- cbind(X_rest[, !colnames(X_rest) %in% NUM_VAR], scaled_vars)

# Training and testing sets
set.seed(123) # For reproducibility
#library(caret)
train_index <- createDataPartition(y_rest, p = 0.8, list = FALSE)
X_rest_train <- X_rest[train_index, ]
y_rest_train <- y_rest[train_index]
X_rest_test <- X_rest[-train_index, ]
y_rest_test <- y_rest[-train_index]

# Gamma GLM
#library(glmnet)
glm_rest <- glmnet(X_rest_train, y_rest_train, family = "gaussian")

# Predictions
pred_rest <- predict(glm_rest, newx = X_rest_test)

# RMSE calculation
RMSE_rest <- sqrt(mean((y_rest_test - pred_rest)^2))
cat("RMSE:", round(RMSE_rest, 3), "\n")
cat("Pred Mean:", round(mean(pred_rest), 3), "\n")
cat("Actual Mean:", round(mean(y_rest_test), 3), "\n")
cat("RMSE / Actual Mean", round(RMSE_rest / mean(y_rest_test), 3), "\n")

############### Combine models

sub_avg <- data.frame(id_policy = test_labels, claim_amount = (predictions_XGB+2*predictions_lasso)/4)
head(sub_avg)

RMSE(claims_amts[-trainIndex],predictions_lasso)
RMSE(claims_amts[-trainIndex],predictions_ridge)
RMSE(claims_amts[-trainIndex],predictions_XGB)
RMSE(claims_amts[-trainIndex],sub_avg$claim_amount)



#### Create something to visualize model fit
library(data.table)
LassoPred <- predict(lasso_mod, test1)
summary(LassoPred)

claims$random_value = sample(1:nrow(claims),nrow(claims),replace = FALSE)/nrow(claims)
validation = claims %>% filter(random_value >= .6 , random_value < .8)
validation$Severity_Estimate = predict(glm,newdata = validation,type = "response")


compare_predictions = function(var,data){
  
  agg = data %>% group_by(!!sym(var)) %>% summarise(Severity = mean(claims$claim_amount),Prediction = mean(validation$Severity_Estimate))
  
  ggplot(agg) +  aes_string(var,"Severity",group = 1) + geom_point() + geom_line(aes(color = "Actual")) +  
    geom_point(aes_string(var,"Prediction",group = 1)) + geom_line(aes_string(var,"Prediction"))
  
}

compare_predictions("drv_age1",test1)
