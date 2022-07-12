# Importing Libraries
install.packages("readr")
library("readr")
install.packages("tidyverse")
library("tidyverse")
install.packages("ggplot2")
library(ggplot2)
install.packages("corrplot")
library(corrplot)
install.packages("ggpubr")
library("ggpubr")
install.packages("data.table")
library(data.table)
library(dplyr)
install.packages('caret')
library(caret)
install.packages("e1071")
library(e1071)
install.packages('rpart')
library(rpart)
install.packages('rpart.plot')
library(rpart.plot)
install.packages('randomForest')
library(randomForest)
install.packages("ROSE")
library(ROSE)

# Clearing workspace and console
rm(list=ls())
cat("\014")

# Setting Working Directory
setwd("C://Users//Poli Ghosh//Documents//Course_Material//Sem 1//R-6356//Project_proposal")

# Setting Seed
set.seed(100) 

# Loading Data
df=read_csv("creditcard.csv", col_names = TRUE)
head(df,10)

# Finding Shape of Data
dim(df)   # The data frame has 284807 no. of rows and 31 no. of columns

# Retrieving Features
names(df) # The features are "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13"
          # "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"
          # "Amount", "Class" 
names(df)=tolower(names(df)) # Changing features to lower case
names(df)

# Finding Datatypes of Features
str(df)
df$class = as.factor(df$class)  # Converting feature class into Categorical datatype
str(df)

# Checking for missing values
df_na_check=is.na(df)
df_na_check
table(df_na_check)["FALSE"]     # There are no N/A values as the number of FALSE is equal to 8829017 which is equal to 
                                # No. of rows * No. of columns = 31*284807

# Exploratory data analysis

# plotting a Boxplot for amount feature
boxplot(df$amount, 
        main = "Transaction Amount in Credit Card Fraud Detection", 
        ylab = "Amount in Dollars", horizontal=TRUE)      # Observed lot of outliers

# Finding Mean and Median 
summary(df)

# plotting a histogram for amount feature
summary(df$amount)
hist(df$amount,main="Transaction Amount in Credit Card Fraud Detection", col="Blue",prob=TRUE,
     xlab="Amount in Dollars")    # Right Skewed histogram

# plotting a histogram for time feature
summary(df$time)
hist(df$time,main="Credit Card Fraud Detection", col="Blue",prob=TRUE,
     xlab="Time in seconds")    

# plotting a scatter plot between amount and class feature
ggplot(data = df) +
  geom_point(mapping = aes(x = amount, y = class))

# plotting a scatter plot between amount and time feature
ggplot(data = df) +
  geom_point(mapping = aes(x = time, y = amount))

# Pearson correlation test
res1 <- cor.test(df$time, df$amount,  method="pearson")
res1   

# In the result above :
# t is the t-test statistic value (t = -5.6553),
# df is the degrees of freedom (df= 284805),
# p-value is the significance level of the t-test (p-value = 1.557e-08).
# conf.int is the confidence interval of the correlation coefficient at 95% (conf.int = [-0.014268414, -0.006924047]);
# sample estimates is the correlation coefficient (Cor.coeff = -0.01059637).

# Interpretation of the result
# The p-value of the test is 1.557e-08, which is less than the significance level alpha = 0.05
# So, we fail to accept the null hypothesis and conclude that true correlation is not equal to 0
# We can conclude that time and amount are not significantly correlated with a correlation coefficient of -0.01059637
# and p-value of 1.557e-08

# Target Variable (Classification)
# Class Distribution
table(df$class)    # No. of 0's - 284315, 1's  - 492

setDT(df)[,100*.N/nrow(df), by = class]       # 0 - 99.8272514% ; 1 - 0.1727486% , indicates data is highly unbalanced

ggplot(df, aes(x=df$class), position = "dodge") +  labs(x='Class',y='No. of Count', title='Predict Variable distribution' )+ geom_bar( width=0.5, fill='lightblue', color='darkblue')

#Oversampling
data_balanced_over <- ovun.sample(class ~ ., data = df, method = "over")$data
table(data_balanced_over$class)
setDT(data_balanced_over)[,100*.N/nrow(data_balanced_over), by = class]  
ggplot(data_balanced_over, aes(x=data_balanced_over$class), position = "dodge") +  labs(x='Class',y='No. of Count', title='Predict Variable distribution' )+ geom_bar( width=0.5, fill='lightblue', color='darkblue')

#Undersampling
data_balanced_under <- ovun.sample(class ~ ., data = df, method = "under")$data
table(data_balanced_under$class)
setDT(data_balanced_under)[,100*.N/nrow(data_balanced_under), by = class] 
ggplot(data_balanced_under, aes(x=data_balanced_under$class), position = "dodge") +  labs(x='Class',y='No. of Count', title='Predict Variable distribution' )+ geom_bar( width=0.5, fill='lightblue', color='darkblue')

#Modelling using Oversampling
#obtain stratified sample
df_sample <- data_balanced_over %>%
  sample_n(size=20000)
head(df_sample,10)

setDT(df_sample)[,100*.N/nrow(df_sample), by = class]    # 0 - 99.815% ; 1 - 0.185%; proportion almost same as that of population

# Removing the feature time from the data frame as not much information can be achieved from it
#sample = subset(sample, select = -c(time) )
nrow(df_sample)

# splitting data into testing and training data sets
# we will first randomly select 4/5 of the rows
train <- sample(1:nrow(df_sample), nrow(df_sample)*(4/5))
typeof(train)
# Use the train index set to split the dataset
#  churn.train for building the model
class.train <- df_sample[train,]   # 16000 rows
nrow(class.train)
#class.train=data.frame(class.train)
typeof(class.train)
#  churn.test for testing the model
class.test <- df_sample[-train,]   # the other 4000 rows
nrow(class.test)
typeof(class.train)
#Split x and y from each test and train dataset
#sapply(df_sample, is.factor)   # Checking if the feature is a factor or not
x.train=class.train[,c(1:29)]
y.train=class.train[,-c(1:29)]

x.test=class.test[,c(1:29)]
y.test=class.test[,-c(1:29)]

#Scaling x features of test and train dataframe
x.train=scale(x.train)
x.test=scale(x.test)

#Base Model 
########### 1. Logistic Regression ########### 
names(class.train)
logit.reg <- glm(class~.,data = class.train, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, class.test, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)

# evaluate classifier on test.df
actual <- class.test$class
predict <- logitPredictClass
cm <- table(predict, actual)
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy    #0.947
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.9154
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity    #0.9781
# FPR
FPR=fp/(fp+tn)
FPR       #0.02184707
# FNR
FNR=fn/(fn+tp)
FNR       #0.08459215
# Precision
Precision = tp / (tp + fp)
Precision     #0.9763695

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_LR = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_LR     # 0.9449064


########### 2. K-Nearest Neighbors ########### 
# Checking distribution of outcome classes -> very few class = "1"
prop.table(table(class.train$class)) * 100  # 0's - 50.425%, 1's - 49.575%
prop.table(table(class.test$class)) * 100   # 0's - 50.35%, 1's - 49.65%
prop.table(table(df_sample$class)) * 100    # 0's - 50.41%, 1's - 49.59%

# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 
# use preProcess to to normalize the predictors
# "center" subtracts the mean; "scale" divides by the standard deviation
knnFit <- train(class~.,data = class.train, method = "knn", trControl = ctrl, preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = 1:10))

# print the knn fit for different values of k
# Kappa is a more useful measure to use on problems that have imbalanced classes.
knnFit
# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)

# Evaluate classifier performance on testing data
actual <- class.test$class
knnPredict <- predict(knnFit, class.test)
cm <- table(knnPredict, actual)
cm 
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy    #0.99975
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #1
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity    #0.9997497
# FPR
FPR=fp/(fp+tn)
FPR      #0.0002502503
# FNR
FNR=fn/(fn+tp)
FNR     #0
# Precision
Precision = tp / (tp + fp)
Precision    #0.8

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_KNN = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_KNN     # 0.8888889

########### 3. Naive Bayes Classifier ########### 
# run naive bayes
fit.nb <- naiveBayes(class~.,data = class.train)
fit.nb

# Evaluate Performance using Confusion Matrix
actual <- class.test$class
# predict class probability
nbPredict <- predict(fit.nb, class.test, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, class.test, type = "class")
cm <- table(nbPredictClass, actual)
cm 

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.977
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #1
# Precision
Precision = tp / (tp + fp)
Precision    #0.04166667
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.976977
# FPR
FPR=fp/(fp+tn)
FPR      #0.02302302
# FNR
FNR=fn/(fn+tp)
FNR      #0

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_NBC = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_NBC     # 0.08


########### 4. Decision Tree ########### 
# grow tree 
fit <- rpart(class~.,data = class.train, 
             method="class", control=rpart.control(xval=0, minsplit=10),
             parms=list(split="gini"))  

fit  # display basic results

# plot a prettier tree using rpart.plot
par("mar")
par(mar=c(1,1,1,1))
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Credit Card Fraud Detection")    
rpart.plot(fit, type = 1, extra = 1, main="Classification Tree for Credit Card Fraud Detection")

# predict class probability
class.pred <- predict(fit, class.train, type="class")
# extract the actual class
class.actual <- class.train$class

# now build the "confusion matrix"
confusion.matrix <- table(class.pred, class.actual) 
cm <- prop.table(confusion.matrix)  
cm

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.9995
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.8148148
# Precision
Precision = tp / (tp + fp)
Precision    #0.88
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.9998122
# FPR
FPR=fp/(fp+tn)
FPR      #0.0001878169
# FNR
FNR=fn/(fn+tp)
FNR      #0.1851852

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_DT1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_DT1     # 0.8461538

#now let us use the hold out data in class.test
class.pred <- predict(fit, class.test, type="class")
class.actual <- class.test$class
confusion.matrix <- table(class.pred, class.actual)
confusion.matrix
cm <- prop.table(confusion.matrix)
cm

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.99975
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #1
# Precision
Precision = tp / (tp + fp)
Precision    #0.8
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.9997497
# FPR
FPR=fp/(fp+tn)
FPR      #0.0002502503
# FNR
FNR=fn/(fn+tp)
FNR      #0

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_DT2 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_DT2     # 0.8888889

#--------------------------------------------------------------------

#Undersampling
data_balanced_under <- ovun.sample(class ~ ., data = df, method = "under")$data
table(data_balanced_under$class)
setDT(data_balanced_under)[,100*.N/nrow(data_balanced_under), by = class] 
ggplot(data_balanced_under, aes(x=data_balanced_under$class), position = "dodge") +  labs(x='Class',y='No. of Count', title='Predict Variable distribution' )+ geom_bar( width=0.5, fill='lightblue', color='darkblue')
nrow(data_balanced_under)  #980
#Modelling using Undersampling
#obtain stratified sample
df_sample <- data_balanced_under %>%
  sample_n(size=800)
head(df_sample,10)

setDT(df_sample)[,100*.N/nrow(df_sample), by = class]    # 0 - 49.75% ; 1 - 50.25%; proportion almost same as that of population

# Removing the feature time from the data frame as not much information can be achieved from it
#sample = subset(sample, select = -c(time) )
nrow(df_sample)

# splitting data into testing and training data sets
# we will first randomly select 4/5 of the rows
train <- sample(1:nrow(df_sample), nrow(df_sample)*(4/5))
typeof(train)
# Use the train index set to split the dataset
#  churn.train for building the model
class.train <- df_sample[train,]   # 16000 rows
nrow(class.train)
#class.train=data.frame(class.train)
typeof(class.train)
#  churn.test for testing the model
class.test <- df_sample[-train,]   # the other 4000 rows
nrow(class.test)
typeof(class.train)
#Split x and y from each test and train dataset
#sapply(df_sample, is.factor)   # Checking if the feature is a factor or not
x.train=class.train[,c(1:29)]
y.train=class.train[,-c(1:29)]

x.test=class.test[,c(1:29)]
y.test=class.test[,-c(1:29)]

#Scaling x features of test and train dataframe
x.train=scale(x.train)
x.test=scale(x.test)

#Base Model 
########### 1. Logistic Regression ########### 
names(class.train)
logit.reg <- glm(class~.,data = class.train, family = "binomial") 
summary(logit.reg)

# use predict() with type = "response" to compute predicted probabilities. 
logitPredict <- predict(logit.reg, class.test, type = "response")
# we choose 0.5 as the cutoff here for 1 vs. 0 classes
logitPredictClass <- ifelse(logitPredict > 0.5, 1, 0)

# evaluate classifier on test.df
actual <- class.test$class
predict <- logitPredictClass
cm <- table(predict, actual)
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy    #0.9387755
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.8977273
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity    #0.9722222
# FPR
FPR=fp/(fp+tn)
FPR       #0.02777778
# FNR
FNR=fn/(fn+tp)
FNR       #0.1022727
# Precision
Precision = tp / (tp + fp)
Precision     #0.9634146

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_LR = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_LR     # 0.9294118


########### 2. K-Nearest Neighbors ########### 
# Checking distribution of outcome classes -> very few class = "1"
prop.table(table(class.train$class)) * 100  # 0's - 49.84375%, 1's - 50.15625 %
prop.table(table(class.test$class)) * 100   # 0's - 49.375%, 1's - 50.625 %
prop.table(table(df_sample$class)) * 100    # 0's - 49.75%, 1's - 50.25%

# 10-fold cross-validation
ctrl <- trainControl(method="cv", number=10) 
# use preProcess to to normalize the predictors
# "center" subtracts the mean; "scale" divides by the standard deviation
knnFit <- train(class~.,data = class.train, method = "knn", trControl = ctrl, preProcess = c("center","scale"),
                tuneGrid = expand.grid(k = 1:10))

# print the knn fit for different values of k
# Kappa is a more useful measure to use on problems that have imbalanced classes.
knnFit
# plot the # of neighbors vs. accuracy (based on repeated cross validation)
plot(knnFit)

# Evaluate classifier performance on testing data
actual <- class.test$class
knnPredict <- predict(knnFit, class.test)
cm <- table(knnPredict, actual)
cm 
# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy    #0.9081633
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.8409091
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity    #0.962963
# FPR
FPR=fp/(fp+tn)
FPR      #0.03703704
# FNR
FNR=fn/(fn+tp)
FNR     #0.1590909
# Precision
Precision = tp / (tp + fp)
Precision    #0.9487179

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_KNN = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_KNN     # 0.8915663

########### 3. Naive Bayes Classifier ########### 
# run naive bayes
fit.nb <- naiveBayes(class~.,data = class.train)
fit.nb

# Evaluate Performance using Confusion Matrix
actual <- class.test$class
# predict class probability
nbPredict <- predict(fit.nb, class.test, type = "raw")
# predict class membership
nbPredictClass <- predict(fit.nb, class.test, type = "class")
cm <- table(nbPredictClass, actual)
cm 

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.9030612
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.8636364
# Precision
Precision = tp / (tp + fp)
Precision    #0.9156627
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.9351852
# FPR
FPR=fp/(fp+tn)
FPR      #0.06481481
# FNR
FNR=fn/(fn+tp)
FNR      #0.1363636

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_NBC = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_NBC     # 0.8888889


########### 4. Decision Tree ########### 
# grow tree 
fit <- rpart(class~.,data = class.train, 
             method="class", control=rpart.control(xval=0, minsplit=10),
             parms=list(split="gini"))  

fit  # display basic results

# plot a prettier tree using rpart.plot
par("mar")
par(mar=c(1,1,1,1))
prp(fit, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, main="Classification Tree for Credit Card Fraud Detection")    
rpart.plot(fit, type = 1, extra = 1, main="Classification Tree for Credit Card Fraud Detection")

# predict class probability
class.pred <- predict(fit, class.train, type="class")
# extract the actual class
class.actual <- class.train$class

# now build the "confusion matrix"
confusion.matrix <- table(class.pred, class.actual) 
cm <- prop.table(confusion.matrix)  
cm

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.9515306
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.9356436
# Precision
Precision = tp / (tp + fp)
Precision    #0.9692308
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.9684211
# FPR
FPR=fp/(fp+tn)
FPR      #0.03157895
# FNR
FNR=fn/(fn+tp)
FNR      #0.06435644

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_DT1 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_DT1     # 0.9521411

#now let us use the hold out data in class.test
class.pred <- predict(fit, class.test, type="class")
class.actual <- class.test$class
confusion.matrix <- table(class.pred, class.actual)
confusion.matrix
cm <- prop.table(confusion.matrix)
cm

# consider class "1" as positive
tp <- cm[2,2]
tn <- cm[1,1]
fp <- cm[2,1]
fn <- cm[1,2]
# accuracy
accuracy=(tp + tn)/(tp + tn + fp + fn)
accuracy     #0.9285714
# TPR = Recall = Sensitivity
Sensitivity = tp/(fn+tp)
Sensitivity   #0.8977273
# Precision
Precision = tp / (tp + fp)
Precision    #0.9404762
# TNR = Specificity
Specificity = tn/(fp+tn)
Specificity   #0.9537037
# FPR
FPR=fp/(fp+tn)
FPR      #0.0462963
# FNR
FNR=fn/(fn+tp)
FNR      #0.1022727

# When using classification models in machine learning, a common metric that we use to assess the quality of the model is the F1 Score.
F1_Score_DT2 = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
F1_Score_DT2     # 0.9186047










