performance0_1046<-read.csv("results0-2720.csv",header=TRUE)
plot(performance0_1046$success_rate)

#Real Linear Affect
summary(lm(success_rate~.,data=performance0_1046))

#Covariance 
View(cor(performance0_1046))

#Graphs

plot(performance0_1046$convolve_shape_x, performance0_1046$success_rate, main = "Convolve Shape X vs Success")
plot(performance0_1046$convolve_shape_y, performance0_1046$success_rate, main = "Convolve Shape Y vs Success")
plot(performance0_1046$binarize_threshold, performance0_1046$success_rate, main = "Binarize Threshold vs Success")
plot(performance0_1046$down_scale_ratio, performance0_1046$success_rate, main = "Down Scale Ratio vs Success")

performance0_1046=performance0_1046[order(performance0_1046$success_rate, decreasing = TRUE),]
View(performance0_1046)
