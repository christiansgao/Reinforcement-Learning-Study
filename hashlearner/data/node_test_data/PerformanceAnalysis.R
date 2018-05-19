library(ggplot2)

performance_df<-read.csv("results0-3465.csv",header=TRUE)
plot(performance_df$success_rate)

#Real Linear Affect
summary(lm(success_rate~.,data=performance_df))

#Covariance 
View(cor(performance_df))

#Graphs

plot(performance_df$convolve_shape_x, performance_df$success_rate, main = "Convolve Shape X vs Success")
plot(performance_df$convolve_shape_y, performance_df$success_rate, main = "Convolve Shape Y vs Success")
plot(performance_df$binarize_threshold, performance_df$success_rate, main = "Binarize Threshold vs Success")
plot(performance_df$down_scale_ratio, performance_df$success_rate, main = "Down Scale Ratio vs Success")

performance_df=performance_df[order(performance_df$success_rate, decreasing = TRUE),]
View(performance_df)

bp <- ggplot(data=performance_df, aes(x=factor(down_scale_ratio), y=success_rate, fill=down_scale_ratio)) +
geom_boxplot() +
labs(title="Categorization Success Rate vs Down Scale Ratio",x ="Down Scale Ratio", y = "Success Rate", fill='Down Scale Ratio') +
  theme(plot.title = element_text(hjust = 0.5))
bp
