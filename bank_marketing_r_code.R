#Bank Marketing Machine Learning Classification Project

#Jennie Hirsch

#Loading Data and Splitting into Training, Validate, Test Sets

setwd("C:/Users/jenni/OneDrive/Documents/Data Science Projects/Bank Marketing/bank")
full_data <- read.csv("bank-full_transform.csv")
set.seed(42); unif_vector <- runif(nrow(full_data))
train_data <- full_data[unif_vector<.8,]
val_data <- full_data[unif_vector>=.8 & unif_vector<.9,]
test_data <-full_data[unif_vector>=.9,]

#Check we have approx. 80-10-10 split
nrow(train_data)/nrow(full_data)
nrow(val_data)/nrow(full_data)
nrow(test_data)/nrow(full_data)

#Logistic Model

formula1 <- as.formula(y~age+I(age^2)+job+marital+education+default+balance + I(balance^2) + I(balance^3)+housing
                       +loan+poutcome + prev_contact + contact)
logistic_model <- glm(formula1,family = binomial,data = train_data)
summary(logistic_model)

#Confusion Matrix and Accuracy on Validation Set
log_val_pred <- predict(logistic_model,newdata = val_data,type = "response")
log_val_class_pred <-ifelse(log_val_pred>0.5,1,0)
val_actual <- val_data$y
log_conf_mat_val <- table(log_val_class_pred,val_actual)
log_conf_mat_val
accuracy <- (log_conf_mat_val[1,1]+log_conf_mat_val[2,2])/sum(log_conf_mat_val)
accuracy

#ROC Chart, AUC on Validation Set
library(ROCR)
log_val_ROC_pr <- prediction(log_val_pred,val_actual)
log_val_ROC_prf <- performance(log_val_ROC_pr,measure = "tpr",x.measure = "fpr")
plot(log_val_ROC_prf)
abline(a = 0, b = 1)
title(main = "Logistic Model ROC Curve on Validation Set")
log_val_auc <- performance(log_val_ROC_pr,measure = "auc")
log_val_auc <-log_val_auc@y.values[[1]]
log_val_auc


# Lift Chart
log_val_lift_prf <- performance(log_val_ROC_pr, measure = "lift", x.measure = "rpp")
plot(log_val_lift_prf)
title(main = "Logistic Model Lift Chart on Validation Set")

# Cumulative Gains Curve

library(ggplot2)
library(scales)

cumulative_gains_data <- function(pred,act){
  table1 = data.frame(pred=pred,act=act)
  table2 = table1[order(-table1$pred),]
  table3 = data.frame(table2,percent_contacted = rep(0,length(act)),percent_of_successes = rep(0,length(act)))
  for(i in 1:length(act)){table3[i,3]=i/length(act) }
  for(i in 1:length(act)){table3[i,4]=sum(table3$act[1:i])/sum(table1$act)}
  data.frame(percent_contacted = table3[,3], percent_of_successes = table3[,4])
}
log_val_gains_data <- cumulative_gains_data(log_val_pred,val_actual)
log_val_gain_chart <- ggplot(log_val_gains_data, aes(x = percent_contacted,y = percent_of_successes))+geom_line(colour = "red")
log_val_gain_chart + ggtitle("Logistic Model Cumulative Gains Curve on Validation Set") + 
    labs(x = "Percent Contacted",y = "Percent of Successes") + scale_y_continuous(labels = percent)+ 
    scale_x_continuous(labels = percent)
log_val_gains_data[length(val_actual)/2,]



#Random Forest with Default Hyperparameters

library(randomForest)
formula2 <- as.formula(response~age+I(age^2)+job+marital+education+default+balance + I(balance^2) + I(balance^3)+housing
                       +loan+poutcome + prev_contact + contact
)
rf_model1 <- randomForest(formula2, data = train_data)
rf_model1
importance(rf_model1)
plot(rf_model1) #note that we don't need 500 trees, but we will keep 500 to be safe

rf1_val_class_pred <- predict(rf_model1, newdata = val_data, type = "response")
rf1_val_pred <- predict(rf_model1,newdata = val_data, type = "prob")[,2]

rf1_val_ROC_pr <- prediction(rf1_val_pred,val_actual)
rf1_val_ROC_prf <- performance(rf1_val_ROC_pr,measure = "tpr",x.measure = "fpr")
plot(rf1_val_ROC_prf)
abline(a = 0, b = 1)
title(main = "Random Forest(#1) Model ROC Curve on Validation Set")
rf1_val_auc <- performance(rf1_val_ROC_pr,measure = "auc")
rf1_val_auc <- rf1_val_auc@y.values[[1]]
rf1_val_auc

rf1_val_gains_data <- cumulative_gains_data(rf1_val_pred,val_actual)
rf1_val_gain_chart <- ggplot(rf1_val_gains_data, aes(x = percent_contacted,y = percent_of_successes))+
       geom_line(colour = "red")
rf1_val_gain_chart + ggtitle("Random Forest(#1) Model Cumulative Gains Curve on Validation Set") +
       labs(x = "Percent Contacted",y = "Percent of Successes")+ scale_y_continuous(labels = percent)+ 
       scale_x_continuous(labels = percent)
rf1_val_gains_data[length(val_actual)/2,]
rf1_conf_mat <- table(rf1_val_class_pred,val_name_actual)
rf1_conf_mat

#Next we will try adjusting the number of variables chosen at each split

val_name_actual <- val_data$response
rf_model <- list(rep(0,14))
rf_val_class_pred <- matrix(rep(0,14*length(val_actual)),ncol = 14)
rf_val_pred <- matrix(rep(0,14*length(val_actual)),ncol = 14)
rf_accuracy <- c(rep(0,14))
rf_val_ROC_pr <- list(rep(0,14))
rf_val_ROC_prf <- list(rep(0,14))
rf_val_auc <- c(rep(0,14))
rf_gains_data <- list(rep(0,14))
for(i in 1:14){
  rf_model[[i]] <- randomForest(formula2,data = train_data, mtry = i)
  rf_val_class_pred[,i] <- predict(rf_model[[i]], newdata = val_data, type = "response")
  rf_val_pred[,i] <- predict(rf_model[[i]],newdata = val_data, type = "prob")[,2]
  rf_accuracy[i] <- sum(rf_val_class_pred[,i]!=val_actual)/length(val_actual) 
  rf_val_ROC_pr[[i]] <- prediction(rf_val_pred[,i],val_actual)
  rf_val_ROC_prf[[i]]<- performance(rf_val_ROC_pr[[i]],measure = "auc")
  rf_val_auc[i] <- rf_val_ROC_prf[[i]]@y.values[[1]]
  rf_gains_data[[i]] <- cumulative_gains_data(rf_val_pred[,i],val_actual)
}

rf_accuracy
rf_val_auc

rf_gains_data2 <- rbind(rf_gains_data[[1]],rf_gains_data[[2]],rf_gains_data[[3]],rf_gains_data[[4]],
                        rf_gains_data[[5]],rf_gains_data[[6]],rf_gains_data[[7]],rf_gains_data[[8]],
                        rf_gains_data[[9]],rf_gains_data[[10]],rf_gains_data[[11]],rf_gains_data[[12]],
                        rf_gains_data[[13]],rf_gains_data[[14]])
rf_gains_data3 <- data.frame(mtry = as.character(rep(1:14, each = length(val_actual))),rf_gains_data2)

ggplot(rf_gains_data3,aes(x = percent_contacted, y = percent_of_successes, color = mtry))+
  geom_line() + ggtitle("Random Forest Models Cumulative Gains Curve on Validation Set") + 
  labs(x = "Percent Contacted",y = "Percent of Successes")+ scale_y_continuous(labels = percent)+ 
   scale_x_continuous(labels = percent)


# The final random forest model chosen will be the one with mtry = 3.

#Try Support Vector Machine (SVM)

library(e1071)
svm_model1 <- svm(formula2, data = train_data, probability = TRUE)
svm_pred <- predict(svm_model1, newdata = val_data, probability = TRUE)
svm_val_pred <- attr(svm_pred,"probabilities")[,2]
svm_val_class_pred <- rep(0,length(val_actual))
for(i in 1:length(val_actual)){
  if(svm_val_pred[i] >= 0.5){svm_val_class_pred[i] = "yes"}
  else {svm_val_class_pred[i] = "no"}
}
svm_conf_mat_val <- table(svm_val_class_pred,val_name_actual)

svm_val_ROC_pr <- prediction(svm_val_pred,val_actual)
svm_val_ROC_prf <- performance(svm_val_ROC_pr,measure = "tpr",x.measure = "fpr")
plot(svm_val_ROC_prf)
abline(a = 0, b = 1)
title(main = "SVM (#1) Model ROC Curve on Validation Set")
svm_val_auc <- performance(svm_val_ROC_pr,measure = "auc")
svm_val_auc <-svm_val_auc@y.values[[1]]
svm_val_auc

svm_val_gains_data <- cumulative_gains_data(svm_val_pred,val_actual)
svm_val_gain_chart <- ggplot(svm_val_gains_data, aes(x = percent_contacted,y = percent_of_successes))+geom_line(colour = "red")
svm_val_gain_chart + ggtitle("SVM (#1) Model Cumulative Gains Curve on Validation Set") + 
  labs(x = "Percent Contacted",y = "Percent of Successes") + scale_y_continuous(labels = percent)+ 
  scale_x_continuous(labels = percent)

#SVM with default parameters doesn't perform well. Adjust Class.weights

wts <- c(0.883,0.117)
names(wts) <- c("yes","no")

svm_model2 <- svm(formula2, data = train_data,class.weights = wts, probability = TRUE)
svm2_pred <- predict(svm_model2, newdata = val_data, probability = TRUE)
svm2_val_pred <- attr(svm2_pred,"probabilities")[,2]
svm2_val_class_pred <- rep(0,length(val_actual))
for(i in 1:length(val_actual)){
  if(svm2_val_pred[i] >= 0.5){svm2_val_class_pred[i] = "yes"}
  else {svm2_val_class_pred[i] = "no"}
}
svm2_conf_mat_val <- table(svm2_val_class_pred,val_name_actual)

svm2_val_ROC_pr <- prediction(svm2_val_pred,val_actual)
svm2_val_ROC_prf <- performance(svm2_val_ROC_pr,measure = "tpr",x.measure = "fpr")
plot(svm2_val_ROC_prf)
abline(a = 0, b = 1)
title(main = "SVM (#2) Model ROC Curve on Validation Set")
svm2_val_auc <- performance(svm2_val_ROC_pr,measure = "auc")
svm2_val_auc <-svm2_val_auc@y.values[[1]]
svm2_val_auc

svm2_val_gains_data <- cumulative_gains_data(svm2_val_pred,val_actual)
svm2_val_gain_chart <- ggplot(svm2_val_gains_data, aes(x = percent_contacted,y = percent_of_successes))+geom_line(colour = "red")
svm2_val_gain_chart + ggtitle("SVM (#2) Model Cumulative Gains Curve on Validation Set") + 
  labs(x = "Percent Contacted",y = "Percent of Successes") + scale_y_continuous(labels = percent)+ 
  scale_x_continuous(labels = percent)

#This model has much better performance, now optimize cost and gamma

cost_vector <- c(0.01,1,100,1000)
gamma_vector <- c(0.1,1,10,100)
svmop_model <- list(rep(0,16))
svmop_pred <- list(rep(0,16))
svmop_val_pred <- matrix(rep(0,16*length(val_actual)),ncol = 16)
svmop_val_ROC_pr <- list(rep(0,16))
svmop_val_ROC_prf <- list(rep(0,16))
svmop_val_auc <- c(rep(0,16))
svmop_gains_data <- list(rep(0,16))

for(i in 1:16){
  svmop_model[[i]] <- svm(formula2,data = train_data, class.weights = wts, cost = cost_vector[ceiling(i/4)] 
                          , gamma = gamma_vector[i - (ceiling(i/4) - 1)*4]  , probability = TRUE)
  svmop_pred[[i]] <- predict(svmop_model[[i]], newdata = val_data, probability = TRUE)
  svmop_val_pred[,i] <- attr(svmop_pred[[i]],"probabilities")[,2]
  svmop_val_ROC_pr[[i]] <- prediction(svmop_val_pred[,i],val_actual)
  svmop_val_ROC_prf[[i]]<- performance(svmop_val_ROC_pr[[i]],measure = "auc")
  svmop_val_auc[i] <- svmop_val_ROC_prf[[i]]@y.values[[1]]
  svmop_gains_data[[i]] <- cumulative_gains_data(svmop_val_pred[,i],val_actual)
}

svmop_gains_data2 <- rbind(svmop_gains_data[[1]], svmop_gains_data[[2]], svmop_gains_data[[3]], svmop_gains_data[[4]],
                           svmop_gains_data[[5]], svmop_gains_data[[6]], svmop_gains_data[[7]], svmop_gains_data[[8]],
                           svmop_gains_data[[9]], svmop_gains_data[[10]], svmop_gains_data[[11]], svmop_gains_data[[12]], 
                           svmop_gains_data[[13]], svmop_gains_data[[14]], svmop_gains_data[[15]], svmop_gains_data[[16]])
svmop_gains_data3 <- data.frame(model = as.character(rep(1:16, each = length(val_actual))),svmop_gains_data2)

ggplot(svmop_gains_data3,aes(x = percent_contacted, y = percent_of_successes, color = model))+geom_line() + 
  ggtitle("SVM Models Cumulative Gains Curve on Validation Set") + labs(x = "Percent Contacted",y = "Percent of Successes") +
   scale_y_continuous(labels = percent)+ 
   scale_x_continuous(labels = percent)

svmop_val_auc

#New honed in grid

cost_vector <- c(0.5,1,10,50)
gamma_vector <- c(0.01,0.05,0.1,0.5)
svmop_model <- list(rep(0,16))
svmop_pred <- list(rep(0,16))
svmop_val_pred <- matrix(rep(0,16*length(val_actual)),ncol = 16)
svmop_val_ROC_pr <- list(rep(0,16))
svmop_val_ROC_prf <- list(rep(0,16))
svmop_val_auc <- c(rep(0,16))
svmop_gains_data <- list(rep(0,16))

for(i in 1:16){
  svmop_model[[i]] <- svm(formula2,data = train_data, class.weights = wts, cost = cost_vector[ceiling(i/4)] 
                          , gamma = gamma_vector[i - (ceiling(i/4) - 1)*4]  , probability = TRUE)
  svmop_pred[[i]] <- predict(svmop_model[[i]], newdata = val_data, probability = TRUE)
  svmop_val_pred[,i] <- attr(svmop_pred[[i]],"probabilities")[,2]
  svmop_val_ROC_pr[[i]] <- prediction(svmop_val_pred[,i],val_actual)
  svmop_val_ROC_prf[[i]]<- performance(svmop_val_ROC_pr[[i]],measure = "auc")
  svmop_val_auc[i] <- svmop_val_ROC_prf[[i]]@y.values[[1]]
  svmop_gains_data[[i]] <- cumulative_gains_data(svmop_val_pred[,i],val_actual)
}

svmop_gains_data2 <- rbind(svmop_gains_data[[1]], svmop_gains_data[[2]], svmop_gains_data[[3]], svmop_gains_data[[4]],
                           svmop_gains_data[[5]], svmop_gains_data[[6]], svmop_gains_data[[7]], svmop_gains_data[[8]],
                           svmop_gains_data[[9]], svmop_gains_data[[10]], svmop_gains_data[[11]], svmop_gains_data[[12]], 
                           svmop_gains_data[[13]], svmop_gains_data[[14]], svmop_gains_data[[15]], svmop_gains_data[[16]])
svmop_gains_data3 <- data.frame(model = as.character(rep(1:16, each = length(val_actual))),svmop_gains_data2)

ggplot(svmop_gains_data3,aes(x = percent_contacted, y = percent_of_successes, color = model))+geom_line() + 
  ggtitle("SVM Models Cumulative Gains Curve on Validation Set") + 
  labs(x = "Percent Contacted",y = "Percent of Successes") +
  scale_y_continuous(labels = percent) + 
  scale_x_continuous(labels = percent)

svmop_val_auc

# Choose the best performing svm model: cost = 10, gamma = 0.01

final_model <- svmop_model[[9]]
fin_mod_pred <- predict(final_model, newdata = test_data, probability = TRUE)
fin_mod_test_pred <- attr(fin_mod_pred,"probabilities")[,2]
fin_mod_class_pred <- rep(0,length(test_actual))
for(i in 1:length(test_actual)){
  if(fin_mod_test_pred[i] >= 0.5){fin_mod_class_pred[i] = "yes"}
  else {fin_mod_class_pred[i] = "no"}
}

test_name_actual <- test_data$response

fin_mod_conf_mat_test <- table(fin_mod_class_pred,test_name_actual)

fin_mod_test_ROC_pr <- prediction(fin_mod_test_pred,test_actual)
fin_mod_test_ROC_prf <- performance(fin_mod_test_ROC_pr,measure = "tpr",x.measure = "fpr")
plot(fin_mod_test_ROC_prf)
abline(a = 0, b = 1)
title(main = "Final SVM Model ROC Curve on Test Set")
fin_mod_test_auc <- performance(fin_mod_test_ROC_pr,measure = "auc")
fin_mod_test_auc <-fin_mod_test_auc@y.values[[1]]
fin_mod_test_auc

fin_mod_test_gains_data <- cumulative_gains_data(fin_mod_test_pred,test_actual)
fin_mod_test_gain_chart <- ggplot(fin_mod_test_gains_data, aes(x = percent_contacted,y = percent_of_successes))+geom_line(colour = "blue")
fin_mod_test_gain_chart + ggtitle("Final SVM Model Cumulative Gains Curve on Test Set") + 
  labs(x = "Percent Contacted",y = "Percent of Successes") + scale_y_continuous(labels = percent)+ 
  scale_x_continuous(labels = percent)

fin_mod_test_gains_data[length(test_actual)/2,]








