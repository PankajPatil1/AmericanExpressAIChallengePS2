rm(list=ls())
setwd("/media/patil/Others/Pankaj Patil/Analytics/HE_Hackathon/Amex2")
library(cowplot)
library(dplyr)
library(caret)
library(mice)
library(xgboost)
library(mice)
library(ggplot2)
library(catboost)
library(pROC)
library(lightgbm)
library(psych)
library(h2o)
library(ggplot2)
######################################################### F j L E j M P O R T ############################################################################=
train <- read.csv("train.csv",header=F)
test <- read.csv("test.csv",header=F)
trainID <- train[,1]
train[,1] <- NULL
numcols <- c(1:10)
catcols <- 11:(dim(train)[2]-1)
target <- train[,dim(train)[2]]
train <- train[,-dim(train)[2]]
id <- test[,1]
test[,1] <- NULL

######################################################### P R E P R O C E S S j N G ############################################################################=
vec <- paste0("V",1:dim(train)[2])
colnames(train) <- vec
colnames(test) <- vec
dt <- rbind(train,test)
trainrows <- nrow(train)
testrows <- nrow(test)
# rm(train)
rm(test)

########################################################### V j S U A L j S A T j O N #############################################################################=
plotter=0
label <- as.factor(target)
if(plotter!=0){
  plot_1 <-   (ggplot(data=train) + geom_histogram(aes(x=train[,1],fill=label)))  # Requires binning
  plot_2 <-  (ggplot(data=train) + stat_density(aes(x=train[,2],fill=label))) # Requires binning
  plot_3 <-  (ggplot(data=train) + stat_density(aes(x=train[,3],fill=label))) #20000 days seems to be most common 
  plot_4 <-  (ggplot(data=train) + stat_density(aes(x=train[,4],fill=label))) # Income outliers still exist 
  plot_5 <-  (ggplot(data=train) + stat_density(aes(x=train[,5],fill=label))) # 0 is most frequent
  plot_6 <-  (ggplot(data=train) + stat_density(aes(x=train[,6],fill=label))) # 0
  plot_7 <-  (ggplot(data=train) + stat_density(aes(x=train[,7],fill=label))) # 0
  plot_8 <-  (ggplot(data=train) + stat_density(aes(x=train[,8],fill=label))) # 98+ is most comon Binning might help
  plot_9 <-  (ggplot(data=train) + stat_density(aes(x=train[,9],fill=label))) # 10 is median
  plot_10 <-  (ggplot(data=train) + stat_density(aes(x=train[,10],fill=label))) # A is most common
  plot_grid(plot_1,plot_2,plot_3,plot_4,plot_5,plot_6,plot_7,plot_8,plot_9,plot_10)
}
# vec <- vector()
# for (j in numcols){
#   colmean <- mean(dt[1:trainrows,j])
#   colsd <- sd(dt[1:trainrows,j])
#   temp <- which((dt[1:trainrows,j]>(colmean+3*colsd) | dt[1:trainrows,j]<(colmean-3*colsd))==1)
#   vec <- append(vec,temp)
# }
# vec <- unique(vec)
# dt <- dt[-vec,]
# trainrows <- trainrows-length(vec)
# vec <- quantile(dt[,7])
# dt[,55] <- sapply(dt[,7],function(z){
#   if(z<vec[2]){z="A"}
#   else if(z<vec[3]){z="B"}
#   else if(z<vec[4]){z="C"}
#   else {z="D"}
# })
# 
# vec <- quantile(dt[,8])
# dt[,56] <- sapply(dt[,8],function(z){
#   if(z<vec[2]){z="A"}
#   else if(z<vec[3]){z="B"}
#   else if(z<vec[4]){z="C"}
#   else {z="D"}
# })
dt[,3] <- log(dt[,3]+1)
dt[,4] <- log(dt[,4]+1)
dt[,7] <- dt[,1]*dt[,2]
# numcols <- append(numcols,55)
preProc <- preProcess(dt[,numcols],method=c("center","scale"))  
dt[,numcols] <- predict(preProc,newdata=dt[,numcols])
for (j in catcols){
  dt[,j] <- as.factor(dt[,j])
}

# dt$V29 <- NULL
sum(!complete.cases(dt))
########################################################### M O D E L L j N G #############################################################################=
train <- dt[1:trainrows,]
test <- dt[(trainrows+1):dim(dt)[1],]
# temp <- train
n <- sample(1:trainrows,size = 50000)
ctrain <- train[n,]
# target2 <- target
ctarget <- target[n]
rm(dt)
nfolds <- 5
folds <- createFolds(ctrain[,1],k=nfolds)

# fit_control <- trainControl(method = "cv",
#                             number = 4,
#                             classProbs = TRUE)
# 
# myGrid <- expand.grid(l2_leaf_reg = c(3,1,5,10),
#                       border_count = c(32,5,10,20,50),
#                       rsm = c(0.1,0.3,0.6,1),
#                       iterations = 100, learning_rate = 0.03, depth = 4)
# classifier <- train(pred_df_train,as.factor(make.names(ctarget[-x])),method=catboost.caret,trControl = fit_control,tuneGrid = myGrid,logging_level="Verbose")
# print(classifier$bestTune)
# myGrid <- expand.grid(l2_leaf_reg = 1,
#                       border_count = 50,
#                       rsm = 1,
#                       iterations = (30:35)*50, learning_rate = 0.03, depth = 4)
# tuner2 <- train(ctrain,as.factor(make.names(ctarget)),method=catboost.caret,trControl = fit_control,tuneGrid = myGrid)
# print(tuner2$bestTune)
# myGrid <- expand.grid(l2_leaf_reg = 1,
#                       border_count = 50,
#                       rsm = 1,
#                       iterations = 1500, learning_rate = c(0.03,0.05,0.08,0.1), depth = c(4,6,8,10))
# tuner3 <- train(train,as.factor(make.names(target)),method=catboost.caret,trControl = fit_control,tuneGrid = myGrid,logging_level="Verbose")
# print(tuner3$bestTune)
# finalGrid <- expand.grid(l2_leaf_reg = 3,
#                          border_count = 32,
#                          rsm = 1,
#                          iterations = 1500, learning_rate = 0.8, depth = 8)



# aucvals <- vector()
vec <- 1:nfolds
# for(j in vec){
#   x <- unlist(folds[j])
# learn_pool <- catboost.load_pool(data=ctrain[-x,],label=ctarget[-x],cat_features = catcols)
# test_pool <- catboost.load_pool(data=ctrain[x,])
# classifier <- catboost.train(learn_pool,params=list(eval_metric="AUC",l2_leaf_reg = 3,
#                                                     border_count = 32,
#                                                     rsm = 1,
#                                                     iterations = 1500, learning_rate = 0.8, depth = 10))
# classifier2 <- catboost.train(learn_pool,params=list(eval_metric="AUC"))
#   y_pred <- catboost.predict(classifier,pool=test_pool)
#   y_pred2 <- catboost.predict(classifier2,pool=test_pool)
#   fpred <- y_pred*0.65 + y_pred2*0.35
#   AUC <- auc(ctarget[x],fpred)
# AUC
#     aucvals[j] <- AUC
# }

aucvals2 <- vector()
ctrain2 <- sapply(ctrain,as.numeric)
ctrain2 <- as.matrix(ctrain2)

for(j in vec){
  x <- unlist(folds[j])
  train_set <- lgb.Dataset(data= (ctrain2[-x,]),label=ctarget[-x])
  test_set <-  as.matrix(ctrain2[x,])
  train_set2 <- xgb.DMatrix(data= (ctrain2[-x,]),label=ctarget[-x])
  learn_pool <- catboost.load_pool(data=ctrain[-x,],label=ctarget[-x],cat_features = catcols)
  test_pool <- catboost.load_pool(data=ctrain[x,])
  classifiercb1 <- catboost.train(learn_pool,params=list(eval_metric="AUC",l2_leaf_reg = 3,
                                                         border_count = 32,
                                                         rsm = 1,
                                                         iterations = 1500, learning_rate = 0.8, depth = 10))
  classifiercb2 <- catboost.train(learn_pool,params=list(eval_metric="AUC"))
  classifiercb3 <- catboost.train(learn_pool,params=list(eval_metric="AUC",l2_leaf_reg = 1,
                                                         border_count = 32,
                                                         rsm = 0.8,
                                                         iterations = 1500, learning_rate = 0.8, depth = 6))
  
  # classifierlgbm <- lightgbm(params=list(objective = "binary",
  #                                         metric = "auc",num_iterations = 1500,learning_rate=0.3),data=train_set,boosting="dart") 
  classifierlgbm2 <- lightgbm(params=list(objective = "binary",
                                          metric = "auc",num_iterations = 1500,learning_rate=0.1),data=train_set) 
  y_predlgbm <- predict(classifierlgbm,test_set)
  y_predlgbm2 <- predict(classifierlgbm2,test_set)
  y_predcb1 <- catboost.predict(classifiercb1,test_pool)
  y_predcb2 <- catboost.predict(classifiercb2,test_pool)
  y_predcb3 <- catboost.predict(classifiercb3,test_pool)
  
  fpred <-  y_predcb1 * 0.1 + y_predcb2*0.1 + y_predcb3*0.1 + y_predlgbm2*0.35 + y_predlgbm*0.35  
  
  AUC <- auc(ctarget[x],fpred)
  AUC
  aucvals2[j] <- AUC
}
# aucvals3 <- vec



rm(train_set)
rm(learn_pool)
rm(test_set)
rm(test_pool)
learn_pool <- catboost.load_pool(data=train,label=target,cat_features = catcols)
test_pool <- catboost.load_pool(data=test,cat_features = catcols)
train2 <- sapply(train,as.numeric)
train2 <- as.matrix(train2)
train_set <- lgb.Dataset(data= (train2),label=target)
test2 <- sapply(test,as.numeric)
test_set <-  as.matrix(test2)
classifiercb1 <- catboost.train(learn_pool,params=list(eval_metric="AUC",l2_leaf_reg = 3,
                                                       border_count = 32,
                                                       rsm = 1,
                                                       iterations = 1500, learning_rate = 0.8, depth = 10))
classifiercb2 <- catboost.train(learn_pool,params=list(eval_metric="AUC"))
classifiercb3 <- catboost.train(learn_pool,params=list(eval_metric="AUC",l2_leaf_reg = 1,
                                                       border_count = 32,
                                                       rsm = 0.8,
                                                       iterations = 1500, learning_rate = 0.8, depth = 6))

classifierlgbm <- lightgbm(params=list(objective = "binary",
                                       metric = "auc",num_iterations = 1500,learning_rate=0.3),data=train_set,boosting="dart")
classifierlgbm2 <- lightgbm(params=list(objective = "binary",
                                        metric = "auc",num_iterations = 1500,learning_rate=0.1),data=train_set) 
y_predlgbm <- predict(classifierlgbm,test_set)
y_predlgbm2 <- predict(classifierlgbm2,test_set)
y_predcb1 <- catboost.predict(classifiercb1,test_pool)
y_predcb2 <- catboost.predict(classifiercb2,test_pool)
y_predcb3 <- catboost.predict(classifiercb3,test_pool)

fpred <-  y_predcb1 * 0.1 + y_predcb2*0.1 + y_predcb3*0.1 + y_predlgbm2*0.35 + y_predlgbm*0.35  


final_results <- data.frame(key = id,score=fpred)
write.csv(final_results,"eny1.csv",row.names = F)
