library(magrittr)
require(randomForest)
library(tidyverse)
library(JOUSBoost)
library(maboost)
library(gbm)
library(onehot)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

get_files = function(folder){
  files <- list.files(path = folder, pattern = "*.csv")
  files <- paste(folder,files,sep="")
  myfiles = lapply(files, read.csv)
  return(myfiles)
}

get_data_from_folder = function(split_ratio, folder, first_half){
  csv_files = get_files(folder)
  predictions = lapply(csv_files,function(df) df$predicted)
  predictions_df = as.data.frame(do.call(cbind,predictions))
  expected = csv_files[[1]]$expected
  total_df = cbind(expected,predictions_df)
  total_df = data.frame(lapply(X=total_df,FUN = as.factor),stringsAsFactors=FALSE)
  split_index = round(nrow(total_df)*split_ratio)
  if(first_half)
    return(total_df[1:split_index,])
  else
    return(total_df[-1:-split_index,])
}

get_binary_data_from_folder = function(split_ratio, folder, first_half){
  total_df = get_data_from_folder(split_ratio, folder, first_half)
  encoder <- onehot(total_df[-1])
  one_hot_df <- data.frame(predict(encoder, total_df))
  one_hot_df$expected = total_df$expected
  return(one_hot_df)
}

train_rf = function(training, ntrees = 100){
  trained_rf=randomForest(expected ~ . , data = training, ntrees=ntrees)
  return(trained_rf)
}

test_rf = function(testing, trained_rf){
  predicted = predict(trained_rf,testing)
  return(predicted)
}

get_performance = function(predicted, expected){
  results = table(predicted,expected)
  successful_predictions = sum(diag(results))
  total_predictions = length(predicted)
  success_rate = successful_predictions/total_predictions
  print(paste("Succes Rate:",success_rate))
}

get_all_performance = function (testing){
  print("getting all initial performance")
  expected = testing$expected
  for(predicted in testing[-1]){
    get_performance(predicted,expected)
  }
}

main = function(){
  
  folder = "./analysis_results_1-50000-60000/"
  training = get_data_from_folder(split_ratio = .5, folder = folder, first_half = TRUE)  
  testing = get_data_from_folder(split_ratio = .5, folder = folder, first_half = FALSE)
  trained_rf = train_rf(training, ntrees = 100)
  predicted = test_rf(testing, trained_rf)
  #get_all_performance(testing)
  get_performance(predicted, testing$expected)
  
}

main - function(){
  
  folder = "./analysis_results_1-50000-60000/"
  training = get_binary_data_from_folder(split_ratio = .5, folder = folder, first_half = TRUE)  
  testing = get_binary_data_from_folder(split_ratio = .5, folder = folder, first_half = FALSE)
  trained_rf = train_rf(training, ntrees = 1000)
  predicted = test_rf(testing, trained_rf)
  #get_all_performance(testing)
  get_performance(predicted, testing$expected)
}

main()

adaboost = function(){
  
  folder = "./analysis_results_1-50000-60000/"
  training = get_binary_data_from_folder(split_ratio = .5, folder = folder, first_half = TRUE)  
  testing = get_binary_data_from_folder(split_ratio = .5, folder = folder, first_half = FALSE)

  gdis<-maboost(expected ~ . , data = training,iter=50,nu=2
                ,breg="l2", type="sparse",bag.frac=1,random.feature=FALSE
                ,random.cost=FALSE, C50tree=FALSE, maxdepth=6,verbose=TRUE)
  predicted= predict(gdis,testing,type="class");

  get_performance(predicted, testing$expected)
  
  gbm_algorithm <- gbm(expected ~ ., data = training, distribution = "adaboost", n.trees = 100)
  gbm_predicted <- predict(gbm_algorithm, test_dataset, n.trees = 5000)
  
  xgb <- xgboost(data = data.matrix(training[,-51]), 
                 label = expected, 
                 eta = 0.1,
                 max_depth = 15, 
                 nround=25, 
                 subsample = 0.5,
                 colsample_bytree = 0.5,
                 seed = 1,
                 eval_metric = "merror",
                 objective = "multi:softprob",
                 num_class = 12,
                 nthread = 3
  )
  
  predicted <- predict(xgb, data.matrix(testing[,-51]))
  get_performance(predicted, testing$expected)
  
}
  