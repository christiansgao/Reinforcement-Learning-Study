library(magrittr)

result_1<-read.csv("results_1.csv",header=TRUE)
result_2<-read.csv("results_2.csv",header=TRUE)
result_3<-read.csv("results_3.csv",header=TRUE)
result_4<-read.csv("results_4.csv",header=TRUE)
result_5<-read.csv("results_5.csv",header=TRUE)

total_training = data.frame(cbind(result_1$predicted,result_2$predicted,result_3$pre,result_4$predicted,result_5$predicted))
names(total_training) = c("predicted_1","predicted_2","predicted_3","predicted_4","predicted_5")

result2_1<-read.csv("results2_1.csv",header=TRUE)
names(result2_1)<-c("expected","predicted_1")
result2_2<-read.csv("results2_2.csv",header=TRUE)
names(result2_2)<-c("expected","predicted_2")
result2_3<-read.csv("results2_3.csv",header=TRUE)
names(result2_2)<-c("expected","predicted_3")

## Performace BaseLine ##

result_table_1 = table(result_1$predicted,result_1$expected)
result_table_2 = table(result_2$predicted,result_2$expected)
result_table_3 = table(result_3$predicted,result_3$expected)
result_table_4 = table(result_4$predicted,result_4$expected)
result_table_5 = table(result_5$predicted,result_5$expected)

sum(diag(result_table_1))
sum(diag(result_table_2))
sum(diag(result_table_3))
sum(diag(result_table_4))
sum(diag(result_table_5))

## Part 1 ##
total_results= data.frame(cbind(as.numeric(result2_1$predicted_1),as.numeric(result2_2$predicted_2),as.numeric(result2_2$expected)))
names(total_results) <- c("predicted_1","predicted_2","expected")

predictions = cbind(result2_1$predicted, result2_1$predicted)

get_rate <- function(result, integer){
  
  predicted = result[result$predicted == integer,]
  predicted_correct = predicted[predicted$expected == integer,]
  
  type_1_error = 1 - nrow(predicted_correct)/ nrow(predicted)
  type_1_error
}

decision_map_1<-data.frame(cbind(sapply(X = 0:9, FUN = get_rate, result=result_1),0:9))
names(decision_map_1)<-c("weight_1","predicted_1")

decision_map_2<-data.frame(cbind(sapply(X = 0:9, FUN = get_rate, result=result_2),0:9))
names(decision_map_2)<-c("weight_2","predicted_2")

total_results = merge(total_results,decision_map_1,by="predicted_1")
total_results = merge(total_results,decision_map_2,by="predicted_2")

extract_predictions <-function(r){
  votes<-as.numeric(rep(0,10))
  votes[as.numeric(r[2])+1]=votes[as.numeric(r[2])+1] + as.numeric(r[5])
  votes[as.numeric(r[1])+1]=votes[as.numeric(r[1])+1] + as.numeric(r[4])
  
  best_vote = which(votes==max(votes))-1
  return(best_vote)
}

total_results$final_predictions = apply(total_results, MARGIN = 1, FUN = extract_predictions)
       
performance_1 = table(total_results$predicted_1,total_results$expected)
performance_2 = table(total_results$predicted_2,total_results$expected)
performance_results = table(total_results$final_predictions,total_results$expected)

sum(diag(performance_results))
sum(diag(performance_1))
sum(diag(performance_2))

## Part 2 ##
get_max = function(l){as.numeric(names(sort(table(as.numeric(l)),decreasing = TRUE)[1]))}
total_training$final_predictions = apply(total_training, MARGIN = 1, FUN = get_max )
performance_results_2 = table(total_training$final_predictions, result_1$expected)
sum(diag(performance_results_2))
