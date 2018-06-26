####Objective####
#Determine which phone, iPhone or Galaxy, is the preferred choice for users so that company can focus their efforts on creating the app on of these two phones. Sentiment analysis data gathering was conducted via AWS.


####Install and Load Packages####
install.packages("caret")
library(caret)
install.packages("rattle") #data mining tool for R
library(rattle)
rattle() #dataset used in rattle must be data.frame
install.packages("corrplot")
library(corrplot)
install.packages("arules")
library(arules)


####Load Data####
getwd()
setwd("C:/Users/Saad/Course4-BigData/250.1 lines/Large Matrix Files")
iPhoneLgMx <- read.csv("iPhoneLargeMatrix.csv")

####Introductory Analysis####
summary(iPhoneLgMx)
str(iPhoneLgMx)
sum(is.na(iPhoneLgMx)) #No blanks in excel file
barplot(iPhoneLgMx$iphoneSentiment, col= "red")
mean(iPhoneLgMx$iphoneSentiment) #average sentiment per line
max(iPhoneLgMx$iphoneSentiment)
min(iPhoneLgMx$iphoneSentiment)
mode(iPhoneLgMx$iphoneSentiment) #Why does this return numeric?
median(iPhoneLgMx$iphoneSentiment)


#Convert dataset into data.frame to conduct analysis in rattle() package
iPhoneLgMx.df <- data.frame(iPhoneLgMx)


#Correlation
corrplot(cor(iPhoneLgMx), order ="hclust")


#Feature Selection: Method One
fitControl <- trainControl(method = "oob", number = 10) #Controls computational nuances of training set through out-of-bag estimate (oob)
rffit <- train(y=iPhoneLgMx$iphoneSentiment, x=iPhoneLgMx, method = "rf", trControl = fitControl)
predictors(rffit)


#Feature Selection: Method Two
descrCor <- cor(iPhoneLgMx)
descrCor
summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .80)
highlyCorDescr


####EXTRA WORK####
sort(highlyCorDescr)
length(highlyCorDescr)
colnames(iPhoneLgMx)[9]

for (i in highlyCorDescr)
{print(colnames(iPhoneLgMx[i]))}
###EXTRA WORK ENDS###


#Create new dataset minus the highly correlated features of the original dataset
iPhoneLgMxNew <- iPhoneLgMx[, -highlyCorDescr]
iPhoneLgMxNew


####Preprocessing####
#Discretize into seven categoriess
disfixed7 <- discretize(iPhoneLgMxNew$iphoneSentiment, "fixed", categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))
disfixed7
#Change attribute values to provide clearer meaning
iPhoneLgMxNew$iphoneSentiment <- factor(iPhoneLgMxNew$iphoneSentiment,levels = c("[-Inf, -50)","[ -50, -10)","[ -10,  -1)","[  -1,   1)","[   1,  10)","[  10,  50)","[  50, Inf]"), labels = c("Very Negative", "Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Positive", "Very Positive"))
summary(disfixed7)
str(disfixed7)


#Add disfixed7 variable to iPhoneLgMxNew data.frame in new column titled iphoneSentiment
iPhoneLgMxNew$iphoneSentiment <- disfixed7
summary(iPhoneLgMxNew$iphoneSentiment) #Number of values in each of seven categories


#Print sentiment count for iPhone
sentimentCount <- summary(disfixed7)
write.csv(sentimentCount, file="sentimentCountiPhone.csv", row.names = TRUE)



####Model Development and Optimization####
trainsample <- iPhoneLgMxNew[0:4000,] #Take first 4000 rows
inTrain <- createDataPartition(y=trainsample$iphoneSentiment, p=.70, list=FALSE)
str(inTrain) #Returns structure of inTrain data
str(trainsample)
training <- trainsample[inTrain,] #Creates training set
testing <- trainsample[-inTrain,] #Creates testing set
nrow(training)#Counts number of rows for training
nrow(testing) #Counts number of rows for testing
set.seed(123)#Set psuedo-random number generator



#Using Caret to run all models
#Set trainControl object: 10 fold cross validation; only used for C5.0 method and knn
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)


#C5.0 Method
C5.0Method <- train(y=training$iphoneSentiment,x=training, method="C5.0", trControl=fitControl)
C5.0Method
predictors(C5.0Method)
C5Predict <- predict(C5.0Method, testing)
C5Predict
postResample(C5Predict, testing$iphoneSentiment) #Returns accuracy and kappa values


#knn Method (did work)
knnMethod <- train(iphoneSentiment~., data = training, method = "knn", trControl=fitControl)
knnMethod
predictors(knnMethod)
knnPredict <- predict(knnMethod, testing)
knnPredict
postResample(knnPredict, testing$iphoneSentiment) #Returns accuracy and kappa values


#svm method
fitControlsvm <- trainControl(method = "repeatedcv", number = 10)
svmMethod <- train(iphoneSentiment~., data = training, method = "svmLinear", scale=FALSE, trControl=fitControlsvm)
svmMethod
predictors(svmMethod)
svmPredict <- predict(svmMethod, testing)
svmPredict
postResample(svmPredict, testing$iphoneSentiment) #Returns accuracy and kappa values


#Random Forest method (WINNER)
fitcontrolRF <- trainControl(method = "oob", number = 10)
rfMethod <- train(iphoneSentiment~., data = training, method="rf", trControl=fitcontrolRF)
rfMethod
predictors(rfMethod)
rfPredict <- predict(rfMethod, testing)
rfPredict
postResample(rfPredict, testing$iphoneSentiment) #Returns accuracy and kappa values

####Analysis on Entire Dataset####
rfPredictNew <- predict(rfMethod, iPhoneLgMxNew) #Random forest model as explained by Incomplete Response dataset
rfPredictNew
postResample(rfPredictNew, iPhoneLgMxNew$iphoneSentiment) #Returns Accuracy and Kappa values of incomplete dataset

