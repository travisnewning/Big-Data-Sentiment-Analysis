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


####Load Dataset####
getwd()
setwd("C:/Users/Saad/Course4-BigData/250.1 lines/Large Matrix Files")
GalaxyLgMx <- read.csv("GalaxyLargeMatrix.csv")


####Introductory Analysis####
summary(GalaxyLgMx)
str(GalaxyLgMx)
sum(is.na(iPhoneLgMx)) #No blanks in excel file
barplot(GalaxyLgMx$galaxySentiment)
mean(GalaxyLgMx$galaxySentiment) #average sentiment per line
max(GalaxyLgMx$galaxySentiment)
min(GalaxyLgMx$galaxySentiment)
mode(GalaxyLgMx$galaxySentiment) #Why does this return numeric?
median(GalaxyLgMx$galaxySentiment)


#Correlation
corrplot(cor(GalaxyLgMx), order ="hclust")



####Feature Selection####
descrCorGal <- cor(GalaxyLgMx)
descrCorGal
summary(descrCorGal[upper.tri(descrCor)])
highlyCorDescrGal <- findCorrelation(descrCorGal, cutoff = .80)
highlyCorDescrGal



#####EXTRA SIDE WORK####
sort(highlyCorDescr)
length(highlyCorDescr)
colnames(iPhoneLgMx)[9]

for (i in highlyCorDescr)
{print(colnames(iPhoneLgMx[i]))}
####END EXTRA SIE WORK####



#Create new dataset minus the highly correlated features of the original dataset
GalaxyLgMxNew <- GalaxyLgMx[, -highlyCorDescrGal]
GalaxyLgMxNew



####Preprocessing####
#Discretize into seven categories
disfixed7Gal <- discretize(GalaxyLgMxNew$galaxySentiment, "fixed", categories= c(-Inf, -50, -10, -1, 1, 10, 50, Inf))
disfixed7Gal
GalaxyLgMxNew$galaxySentiment <- factor(GalaxyLgMxNew$galaxySentiment,levels = c("[-Inf, -50)","[ -50, -10)","[ -10,  -1)","[  -1,   1)","[   1,  10)","[  10,  50)","[  50, Inf]"), labels = c("Very Negative", "Negative", "Somewhat Negative", "Neutral", "Somewhat Positive", "Positive", "Very Positive"))
summary(disfixed7Gal)
str(disfixed7Gal)
GalaxyLgMxNew$galaxySentiment <- disfixed7Gal
summary(GalaxyLgMxNew$galaxySentiment) #Number of values in each of seven categories


#Print sentiment count for Galaxy
sentimentCountGal <- summary(disfixed7Gal)
write.csv(sentimentCountGal, file="sentimentCountGalaxy.csv", row.names = TRUE)


####Model Development and Optimization####
trainsampleGal <- GalaxyLgMxNew[0:4000,] #Take first 4000 observations from dataset
inTrain <- createDataPartition(y=trainsampleGal$galaxySentiment, p=.70, list=FALSE)
str(inTrain) #Returns structure of inTrain data
str(trainsample)
trainingGal <- trainsampleGal[inTrain,] #Creates training set
testingGal <- trainsampleGal[-inTrain,] #Creates testing set
nrow(trainingGal)#Counts number of rows for training
nrow(testingGal) #Counts number of rows for testing
set.seed(123)#Set psuedo-random number generator



#Using Caret to run all models
#Set trainControl object: 10 fold cross validation; only used for C5.0 method and knn
fitControlGal <- trainControl(method = "repeatedcv", number = 10, repeats = 10)


#C5.0 Method
C5.0MethodGal <- train(y=trainingGal$galaxySentiment,x=trainingGal, method="C5.0", trControl=fitControlGal)
C5.0MethodGal
predictors(C5.0MethodGal)
C5PredictGal <- predict(C5.0MethodGal, testingGal)
C5PredictGal
postResample(C5PredictGal, testingGal$galaxySentiment) #Returns accuracy and kappa values


#knn Method (did work)
knnMethodGal <- train(galaxySentiment~., data = trainingGal, method = "knn", trControl=fitControlGal)
knnMethodGal
predictors(knnMethodGal)
knnPredictGal <- predict(knnMethodGal, testingGal)
knnPredictGal
postResample(knnPredictGal, testingGal$galaxySentiment) #Returns accuracy and kappa values


#svm method
fitControlsvmGal <- trainControl(method = "repeatedcv", number = 10)
svmMethodGal <- train(galaxySentiment~., data = trainingGal, scale = FALSE, method = "svmLinear", trControl=fitControlsvmGal)
svmMethodGal
predictors(svmMethodGal)
svmPredictGal <- predict(svmMethodGal, testingGal)
svmPredictGal
postResample(svmPredictGal, testingGal$galaxySentiment) #Returns accuracy and kappa values


#Random Forest method (WINNER Galaxy)
fitcontrolRFGal <- trainControl(method = "oob", number = 10)
rfMethodGal <- train(galaxySentiment~., data = trainingGal, method="rf", trControl=fitcontrolRFGal)
rfMethodGal
predictors(rfMethodGal)
rfPredictGal <- predict(rfMethodGal, testingGal)
rfPredictGal
postResample(rfPredictGal, testingGal$galaxySentiment) #Returns accuracy and kappa values


####Analysis on Entire Dataset####
rfPredictNewGal <- predict(rfMethodGal, GalaxyLgMxNew) #Random forest model as explained by Incomplete Response dataset
rfPredictNewGal
postResample(rfPredictNewGal, GalaxyLgMxNew$galaxySentiment) #Returns Accuracy and Kappa values of incomplete dataset
