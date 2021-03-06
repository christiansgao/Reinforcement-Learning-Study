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

EXPECTED_NAMES = c("expected.0","expected.1", "expected.2" ,"expected.3" ,"expected.4", "expected.5", "expected.6" ,"expected.7" ,"expected.8", "expected.9")

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

get_binary_data_total = function(split_ratio, folder, first_half){
  total_df = get_data_from_folder(split_ratio, folder, first_half)
  encoder <- onehot(total_df)
  one_hot_df <- data.frame(predict(encoder, total_df))
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

xg_boost_train = function(training,nround = 10){
  raw_training = data.matrix(training[names(training) != EXPECTED_NAMES])
  xg_list = NULL
  i = 1
  
  for(expected_name in EXPECTED_NAMES){
    label = as.numeric(unlist(training[expected_name]))
    bstSparse <- xgboost(data = raw_training, label = label, max.depth = 5, eta = 1, nthread = 2, nround = nround, objective = "binary:logistic")
    xg_list[i] = list(bstSparse)
    i=i+1
  }
  
  xg_list
}

xgb_predict = function(testing,xg_list){
  extract_prediction = function(testing_xgb,raw_testing){
    gbm_predicted <- predict(testing_xgb, raw_testing)
    #gbm_predicted <- plogis(2*gbm_predicted)
    
    return(unlist(gbm_predicted))
  }
  
  raw_testing = data.matrix(testing[names(training) != EXPECTED_NAMES])
  
  probs = lapply(xg_list, FUN=extract_prediction,raw_testing=raw_testing)
  probs_df = as.data.frame(do.call(cbind,probs))
  
  apply(probs_df,MARGIN = 2,FUN=sd)
  probs_df=scale(probs_df, scale=TRUE)
  colMeans(probs_df)
  
  predicted = apply(MARGIN = 1, probs_df, FUN = function(row_df) which(row_df == max(row_df))[[1]])-1
  predicted
}

main = function(){
  
  folder = "./analysis_results_1-50000-60000/"
  training = get_binary_data_total(split_ratio = .5, folder = folder, first_half = TRUE)  
  testing = get_binary_data_total(split_ratio = .5, folder = folder, first_half = FALSE)
  expected = get_data_from_folder(split_ratio = .5, folder = folder, first_half = FALSE)$expected
  xg_list = xg_boost_train(training, nround = 8)
  predicted = xgb_predict(testing, xg_list)
  #get_all_performance(testing)
  get_performance(predicted, expected)
}

main()
1-0.964
  