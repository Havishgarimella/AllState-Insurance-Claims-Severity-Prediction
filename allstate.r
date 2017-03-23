#install necessary libraries

library(e1071)
library(dplyr)
library(reshape2)
library(ModelMetrics)


#upload the data

setwd("C:/Users/hvg15/Desktop/R/Allstate")

allstate_train <- read.csv("train.csv", header = TRUE)

allstate_test <- read.csv("test.csv", header = TRUE)
nrow(allstate_train)

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


#intial linear regression model
initial.large.model = lm(loss ~ . , data = allstate_train)

initial.large.model

summary(initial.large.model)

rmse(initial.large.model$residuals)

#transformation of prediction variable loss with log10 + 200
allstate_train$loss <- log10(allstate_train$loss + 200)

#linear regression after log transformation
second.large.model = lm(loss ~ . , data = allstate_train)
summary(second.large.model)

predict_second.large.model <- predict(second.large.model, newdata = allstate_test)






