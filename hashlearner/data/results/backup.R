library(magrittr)
require(randomForest)

result_1<-read.csv("results_1.csv",header=TRUE)
result_2<-read.csv("results_2.csv",header=TRUE)
result_3<-read.csv("results_3.csv",header=TRUE)
result_4<-read.csv("results_4.csv",header=TRUE)
result_5<-read.csv("results_5.csv",header=TRUE)

total_training = data.frame(cbind(result_1$predicted,result_2$predicted,result_3$pre,result_4$predicted,result_5$predicted,as.factor(result_5$expected)))
names(total_training) = c("predicted_1","predicted_2","predicted_3","predicted_4","predicted_5","expected")



result2_1<-read.csv("results2_1.csv",header=TRUE)
names(result2_1)<-c("expected","predicted_1")
result2_2<-read.csv("results2_2.csv",header=TRUE)
names(result2_2)<-c("expected","predicted_2")
result2_3<-read.csv("results2_3.csv",header=TRUE)
names(result2_3)<-c("expected","predicted_3")

test_results= data.frame(cbind(as.character(result2_1$predicted_1),as.character(result2_2$predicted_2),as.character(result2_3$predicted_3)))
names(test_results) <- c("predicted_1","predicted_2","predicted_3")

## RF ##

test_random_forest=randomForest(expected ~ predicted_1 + predicted_2 + predicted_3 , data = total_training, ntrees=100)
plot(test_random_forest)

predicted = predict(test_random_forest,test_results)
table(predicted,result2_3$expected)

?randomForest









