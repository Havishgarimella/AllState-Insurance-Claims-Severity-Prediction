#install necessary libraries

library(e1071)
library(dplyr)
library(reshape2)
library(ModelMetrics)
install.packages("matrix")
library(Matrix)
library(data.table)
install.packages("xgboost")
library(xgboost)


#upload the data
setwd("C:/Users/hvg15/Desktop/R/Allstate")

allstate_train <- read.csv("train.csv", header = TRUE)

allstate_test <- read.csv("test.csv", header = TRUE)

allstate_train_nrow <- nrow(allstate_train)

ncol(allstate_train)


### These commands let you see a snippet of the data.
str(allstate_train)

head(allstate_train)

tail(allstate_train)

#Let's see if there are missing values in data
sum(is.na(allstate_train))

#Plot dependent variable loss
hist(allstate_train$loss, col = "grey")

#get all the five points summary of loss
summary(allstate_train$loss)

boxplot(allstate_train$loss, col = "grey")

#plots of all continuos variables in initial-plots.pdf
pdf(file="initial-plots.pdf")

for (i in 118:131) 
  hist(allstate_train[,i], xlab=colnames(allstate_train)[i], data = allstate_train)
dev.off()

#calculate skewness of all numerical variables
sapply(allstate_train[118:132], function(x) skewness(x))


 
#/correlations between numeric variables after deleting duplicate rows in list

allstate_train_cor <- as.matrix(cor(allstate_train[118:131]))

allstate_train_cor_melt <- arrange(melt(allstate_train_cor), -abs(value))

allstate_train_cor_melt <- subset(allstate_train_cor_melt, value > .5)

allstate_train_cor_melt <- subset(allstate_train_cor_melt,allstate_train_cor_melt[,1]!= allstate_train_cor_melt[,2] )

cols <- c(1,2)

new_allstate_train_cor_melt <- allstate_train_cor_melt[,cols]

for (i in 1:nrow(allstate_train_cor_melt))
{
  new_allstate_train_cor_melt[i, ] = sort(allstate_train_cor_melt[i, cols])
}

allstate_train_cor_melt<-allstate_train_cor_melt[!duplicated(new_allstate_train_cor_melt),]

#cont12 and cont11 show high correlatins one should be removed
#cont9 and cont1 show high correlations one can be removed safely


#data preparation for modelling
#transformation of prediction variable loss with log10 + 200
allstate_train$loss <- log10(allstate_train$loss + 200)
model_allstate_trainid <- allstate_train$id
model_allstate_trainloss <- allstate_train$loss
model_allstate_testid <- allstate_test$id
allstate_train[,c("id","loss")] <- NULL
allstate_test[,"id"] <-  NULL
allstate_full<- rbind(allstate_train,allstate_test)
allstate_full_nrow <- nrow(allstate_full)


#one hot coding
features<- names(allstate_full)
factor_features<-vector()
for(feature in features){
  if(class(allstate_full[,feature])=="factor"){
    allstate_full[,feature]<- as.numeric(allstate_full[,feature])-1
  }
}

model_allstate_full <- allstate_full
model_allstate_full[,c("cont12","cont9")] <- NULL
model_allstate_train <- model_allstate_full[1:allstate_train_nrow,]
model_allstate_test <- model_allstate_full[(allstate_train_nrow+1):allstate_full_nrow,]


#intial linear regression model
initial.large.model = lm(model_allstate_trainloss ~ . , data = model_allstate_train)

summary(initial.large.model)
predict_initial.large.model <- predict(initial.large.model, newdata = model_allstate_test)

#what is present in initial.large.model
names(initial.large.model)

#fitted values in the initial.large.model represents predicted values
#plot of observed and predicted values
plot(model_allstate_trainloss,initial.large.model$fitted.values, pch = 19)

training.rmse    = sqrt(   mean( (model_allstate_trainloss - initial.large.model$fitted.values)^2 )   )
training.rmse

#training rmse 0.2246233
write.csv(cbind(model_allstate_testid, predict_initial.large.model),file = "submission.csv")

#reverting the logarthamic transformation
predict_initial.large.model<- (10^predict_initial.large.model)-200


#second linear regression model without log transformation of loss varibale
second.large.model = lm(model_allstate_trainloss ~ . , data = model_allstate_train)

summary(second.large.model)
predict_second.large.model <- predict(second.large.model, newdata = model_allstate_test)

plot(model_allstate_trainloss,second.large.model$fitted.values, pch = 19)

training.rmse_second    = sqrt(   mean( (model_allstate_trainloss - second.large.model$fitted.values)^2 )   )
training.rmse_second

#training rmse second is 2088.88


#Xgboost
xgb_matrix_train_AllState<- xgb.DMatrix(as.matrix(model_allstate_train),label=model_allstate_trainloss)
xgb_matrix_test_AllState<- xgb.DMatrix(as.matrix(model_allstate_test))
xgb_parameters_AllState<- list(
                                objective='reg:linear',
                                seed=2000,
                                colsample_bytree=0.7,
                                subsample=0.7,
                                eta=0.3,
                                max_depth=8,
                                num_parallel_tree=1,
                                min_child_weight=1,
                                base_score=7)


xg_eval_mae<- function(yhat,xgb_matrix_train_AllState){
  y=getinfo(xgb_matrix_train_AllState,"label")
  err= mae(exp(y),exp(yhat))
  return(list(metric="error",value=err))
  
}
res = xgb.cv(xgb_parameters_AllState,
             xgb_matrix_train_AllState,
             nrounds=2550,
             nfold=10,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             maximize=FALSE)


best_nrounds= res$best_iteration
cv_mean<- res$evaluation_log$train_error_mean[best_nrounds]
cv_std<- res$evaluation_log$train_error_std[best_nrounds]
cat(paste0('CV-Mean: ',cv_mean,' ', cv_std))

gbdt<- xgb.train(xgb_parameters_AllState,xgb_matrix_train_AllState,best_nrounds)
submission= predict(gbdt,xgb_matrix_test_AllState)


