#Part-A(Reading the data)
#Data File: "healthcare-dataset-stroke-data.csv"
#import the healthcare-dataset-stroke-data datafile and assign it to myData
#type your own path in here while running the code
myData <- read.csv("/Users/Srujan/Desktop/R Project/healthcare-dataset-stroke-data.csv")
View(myData)
summary(myData)
#Part-B
#Remove any missing values in the dataset.
#To detect the missing values in a data set, we use is.na()
sum(is.na(myData))
#As there are no missing values in the dataset, the data remains the same.
table(myData$gender)
#In the gender column remove values other than Male or  Female
#(1 observation has other so we will eliminate it as we are focusing only on male and female)
myData <- myData[-which(myData$gender!='Male' &  myData$gender!='Female'),]
#In bmi column we have many entries with N/A values.So we will replace them with mean bmi of the dataset.
#To do this I converted bmi column values to numeric so N/A got converted to NA
myData$bmi <- as.numeric(myData$bmi)
#Now,I replaced NA values with the mean of the bmi column
myData$bmi[is.na(myData$bmi)] <- mean(myData$bmi,na.rm=TRUE)
#We will also remove id column because its not required for the prediction of heart stroke.
myData$id <- NULL
#We will convert 1's and 0's in the colums stroke,hypertension and heart disease to "Yes" and "NO" for easier understanding.
myData$hypertension <- factor(ifelse(myData$hypertension == 1, "Yes", "No"))
myData$heart_disease <- factor(ifelse(myData$heart_disease == 1, "Yes", "No"))
myData$stroke <- factor(ifelse(myData$stroke == 1, "Yes", "No"))

#In our dataset, we have 3 numeric variables.So we check for outliers using boxplot.
boxplot(myData$age,main="Distribution of age",col="gold")
boxplot(myData$avg_glucose_level,main="Distribution of avg glucose level",col="red")
boxplot(myData$bmi,main="Distribution of body mass index",col="blue")
#From the graphs in the output,avg_glucose_level and bmi have outliers. Now we need to remove them.
#To treat outliers, we use the out parameter in the boxplot function
avg_glucose_level_outliers <- boxplot(myData$avg_glucose_level)$out
bmi_outliers <- boxplot(myData$bmi)$out
#Now we replace outliers with NA values.
myData$bmi <- ifelse(myData$bmi %in% bmi_outliers, NA, myData$bmi)
myData$avg_glucose_level <- ifelse(myData$avg_glucose_level %in% avg_glucose_level_outliers, NA, myData$avg_glucose_level)
#NA values are now omitted which indirectly has removed the outliers
myData2 <- na.omit(myData)
table(myData2$stroke)
#Now let's draw few graphs to get few insights from the dataset using histograms and stacked column charts.
hist(myData2$age,main="Frequency distribution of age",xlab="Age",col="greenyellow")
hist(myData2$avg_glucose_level,main="Frequency distribution of average glucose level",xlab="Glucose level",col="yellow")
#Here in the barplot I got some problem because legend is intersecting the graph and we were unable to read the graph properly.
#I checked few sites and got the help from a youtube video.Here is the reference of the link:
#https://www.youtube.com/watch?v=ClFNsZaVNFk
mytable <- table(myData2$stroke,myData2$gender)
par(mar = c(5.1,4.1,4.1,6.1))
barplot(mytable, col=c('blue','red'), legend=rownames(mytable),args.legend=list(title="Stroke",x = "topright",inset= c(-0.15,0)), xlab="Gender",
        ylab="Count")
mytable2 <- table(myData2$stroke,myData2$hypertension)
barplot(mytable2, col=c('orange','green'), legend=TRUE,args.legend=list(title="Stroke"), xlab="Hypertension",
        ylab="Count")
mytable3 <- table(myData2$stroke,myData2$heart_disease)
barplot(mytable3, col=c('lightblue','lightpink'), legend=TRUE,args.legend=list(title="Stroke"), xlab="Heart Disease",
        ylab="Count")
mytable4 <- table(myData2$stroke,myData2$Residence_type)
par(mar = c(5.1,4.1,4.1,6.1))
barplot(mytable4, col=c('violet','black'), legend=TRUE,args.legend=list(title="Stroke",x = "topright",inset= c(-0.15,0)), xlab="Residence type",
        ylab="Count")
mytable5 <- table(myData2$stroke,myData2$age)
par(mar = c(5.1,4.1,4.1,6.1))
barplot(mytable5, col=c('blue','red'), legend=TRUE,args.legend=list(title="Stroke",x = "topright",inset= c(-0.15,0)),xlab="Age",
        ylab="Count")
mytable6 <- table(myData2$stroke,myData2$work_type)
par(mar = c(5.1,4.1,4.1,6.1))
barplot(mytable6, col=c('pink','blue'), legend=TRUE,args.legend=list(title="Stroke",x = "topright",inset= c(-0.15,0)),xlab="Work type",
        ylab="Count")
mytable7 <- table(myData2$stroke,myData2$smoking_status)
par(mar = c(5.1,4.1,4.1,6.1))
barplot(mytable7, col=c('orange','greenyellow'), legend=TRUE,args.legend=list(title="Stroke",x = "topright",inset= c(-0.15,0)),xlab="Smoking status",
        ylab="Count")
#After the graphical interpretations we will make the stroke column in the form of binary to use logistic regression
myData2$stroke <- ifelse(myData2$stroke=="Yes",1,0)
table(myData2$stroke)
#We partition the sample into training and validation sets, labeled TData and
#VData, respectively.
#TData <- myData2[1098:4390,]
#VData <- myData2[1:1097,]
#dt = sort(sample(nrow(myData2), nrow(myData2)*.75))
#TData<-myData2[dt,]
#VData<-myData2[-dt,]
install.packages("caret")
library(caret)
#We use the set.seed command to set the random seed to 1, thus
#generating the same partitions as in this example. 
set.seed(1)
Index <- createDataPartition(myData2$stroke, p=0.75, list=FALSE)
TData <- myData2[Index,]
VData <- myData2[-Index,]
#We use the training set, TData, to estimate Model 1 with all variables 
Model1 <- glm(stroke~gender+age+hypertension+heart_disease+ever_married+work_type+
                Residence_type+avg_glucose_level+bmi+smoking_status,family=binomial,data=TData)
summary(Model1)
#Now we change the predictor variables and try different possible models and choose the best model out of them.
Model2 <- glm(stroke~gender+age+hypertension+avg_glucose_level+smoking_status+bmi,family=binomial,data=TData)
summary(Model2)
Model3 <- glm(stroke~age+heart_disease+work_type+Residence_type+bmi ,family=binomial,data=TData)
summary(Model3)
Model4 <- glm(stroke~age+hypertension+avg_glucose_level+ever_married ,family=binomial,data=TData)
summary(Model4)
Model5 <- glm(stroke~age+avg_glucose_level+ever_married  ,family=binomial,data=TData)
summary(Model5)
#We use the estimates to make predictions for Vdata and then
# use predict function to compute the predicted probability for Logistic model. 
# Enter:
pHat1 <- predict(Model1, VData, type = "response")
pHat2 <- predict(Model2, VData, type = "response")
pHat3 <- predict(Model3, VData, type = "response")
pHat4 <- predict(Model4, VData, type = "response")
pHat5 <- predict(Model5, VData, type = "response")

#We use the ifelse function to convert probabilities into binary, 1 or 0, values,
#using a cutoff of 0.5. Enter:

yHat1 <- ifelse(pHat1 >= 0.1, 1,0)
yHat2 <- ifelse(pHat2 >= 0.1, 1,0)
yHat3 <- ifelse(pHat3 >= 0.1, 1,0)
yHat4 <- ifelse(pHat4 >= 0.1, 1,0)
yHat5 <- ifelse(pHat5 >= 0.1, 1,0)

yTP1 <- ifelse(yHat1 == 1 & VData$stroke == 1, 1, 0)
yTN1 <- ifelse(yHat1 == 0 & VData$stroke == 0, 1, 0)
yTP2 <- ifelse(yHat2 == 1 & VData$stroke == 1, 1, 0)
yTN2 <- ifelse(yHat2 == 0 & VData$stroke == 0, 1, 0)
yTP3 <- ifelse(yHat3 == 1 & VData$stroke == 1, 1, 0)
yTN3 <- ifelse(yHat3 == 0 & VData$stroke == 0, 1, 0)
yTP4 <- ifelse(yHat4 == 1 & VData$stroke == 1, 1, 0)
yTN4 <- ifelse(yHat4 == 0 & VData$stroke == 0, 1, 0)
yTP5 <- ifelse(yHat5 == 1 & VData$stroke == 1, 1, 0)
yTN5 <- ifelse(yHat5 == 0 & VData$stroke == 0, 1, 0)
sprintf("Accuracy measure for Model1 = %f",100*mean(VData$stroke == yHat1))
sprintf("Sensitivity for Model1 = %f",100*(sum(yTP1)/sum(VData$stroke==1)))
sprintf("Specificity for Model1 = %f",100*(sum(yTN1)/sum(VData$stroke==0)))
sprintf("Accuracy measure for Model2 = %f",100*mean(VData$stroke == yHat2))
sprintf("Sensitivity for Model2 = %f",100*(sum(yTP2)/sum(VData$stroke==1)))
sprintf("Specificity for Model2 = %f",100*(sum(yTN2)/sum(VData$stroke==0)))
sprintf("Accuracy measure for Model3 = %f",100*mean(VData$stroke == yHat3))
sprintf("Sensitivity for Model3 = %f",100*(sum(yTP3)/sum(VData$stroke==1)))
sprintf("Specificity for Model3 = %f",100*(sum(yTN3)/sum(VData$stroke==0)))
sprintf("Accuracy measure for Model4 = %f",100*mean(VData$stroke == yHat4))
sprintf("Sensitivity for Model4 = %f",100*(sum(yTP4)/sum(VData$stroke==1)))
sprintf("Specificity for Model4 = %f",100*(sum(yTN4)/sum(VData$stroke==0)))
sprintf("Accuracy measure for Model5 = %f",100*mean(VData$stroke == yHat5))
sprintf("Sensitivity for Model5 = %f",100*(sum(yTP5)/sum(VData$stroke==1)))
sprintf("Specificity for Model5 = %f",100*(sum(yTN5)/sum(VData$stroke==0)))
#By looking at all the 5 models above, Model5 is the more accurate one
#because it has low AIC value and a higher accuracy also.

#Classification tree model
#Install and load the caret, gains, rpart, rpart.plot, and pROC packages
#using the following commands. Enter:

install.packages("gains")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("pROC")
library(gains)
library(rpart)
library(rpart.plot)
library(pROC)
options(scipen=999)# to avoid scientific notation
myData2$stroke <- as.factor(myData2$stroke)
set.seed(1)
#We use the createDataPartition function to partition the data into training (75%)
#and validation (25%) data sets.
#I tried doing with 70:30 split but I did get the tree correctly because all the 1's 
#are in training set and the sensitivity became 0 

myIndex <- createDataPartition(myData2$stroke, p=0.75, list=FALSE)
trainSet <- myData2[myIndex,]
validationSet <- myData2[-myIndex,]

#We use the rpart function to generate the default classification tree,
#default_tree.
set.seed(1)
default_tree <- rpart(stroke ~ ., data = trainSet, method = "class")
summary(default_tree)
#To view the classification tree visually, we use the prp function.
prp(default_tree, type = 1, extra = 1, under = TRUE)
#We use the set.seed command to set the random seed to 1, thus
#generating the same partitions as in this example.
set.seed(1)
#Now,full tree is created using rpart.
full_tree <- rpart(stroke ~ ., data = trainSet, method = "class", cp = 0, minsplit = 2, minbucket = 1)
#To view the classification tree visually we use the prp function.
prp(full_tree, type = 1, extra = 1, under = TRUE)
printcp(full_tree)
#Now we use prune function to get the pruned tree taking the cp value of the minimum error tree
#pruned_tree <- prune(full_tree, cp = 0.0161290 )
pruned_tree <- prune(full_tree, cp = 0.0100806  )
prp(pruned_tree, type = 1, extra = 1, under = TRUE)
#We predict the class memberships of the observations in the validation
#data set using the predict function.
predicted_class <- predict(pruned_tree, validationSet, type = "class")
#We use the confusionMatrix function to produce the confusion matrix
#and various performance measures.
confusionMatrix(predicted_class, validationSet$stroke, positive = "1")
predicted_prob <- predict(pruned_tree, validationSet, type= 'prob')
#To construct a confusion matrix using the new cutoff value of 0.037, we
#use the ifelse function to determine the class memberships and convert
#them into text labels of 1s and 0s
confusionMatrix(as.factor(ifelse(predicted_prob[,2]>0.037, '1', '0')), validationSet$stroke, positive = '1')
validationSet$stroke <- as.numeric(as.character(validationSet$stroke))
gains_table <- gains(validationSet$stroke, predicted_prob[,2])
gains_table

# Lift Chart
plot(c(0, gains_table$cume.pct.of.total*sum(validationSet$stroke)) ~ c(0, gains_table$cume.obs), xlab = '# of cases', ylab = "Cumulative", type = "l")
lines(c(0, sum(validationSet$stroke))~c(0, dim(validationSet)[1]), col="red", lty=2)

# Decile Chart 
barplot(gains_table$mean.resp/mean(validationSet$stroke), names.arg=gains_table$depth, xlab="Percentile", ylab="Lift", ylim=c(0, 12.0), main="Decile-Wise Lift Chart")

# ROC chart
roc_object <- roc(validationSet$stroke, predicted_prob[,2])
plot.roc(roc_object)
auc(roc_object)

# #Here is the code for kmeans clustering if required. You can select all the commenting
# #lines and press ctrl+shift+c to run and check the code.I had made this code in comments
# #because we can't interpret anythis by clustering for this dataset.
# #kmeansclustering
# myData3 <- myData2[c(1,3,9,10)]
# suppressWarnings(RNGversion("3.5.3"))
# #install cluster package
# install.packages("cluster")
# library(cluster)
# #Standardize variables using scale
# myData3 <- scale(myData3[, 2:4])
# set.seed(1)
# kResult <- pam(myData3, k = 4)
# 
# # Summarize results
# summary(kResult)
# 
# # Part -d: Use plot function to visualize / show clusters
# plot(kResult)
# 
# #kmeansclustering
# myData3 <- myData2[c(1,3,9,10)]
# suppressWarnings(RNGversion("3.5.3"))
# #install cluster package
# install.packages("cluster")
# library(cluster)
# #Standardize variables using scale
# myData3 <- scale(myData3[, 2:4])
# set.seed(1)
# kResult <- pam(myData3, k = 4)
# 
# # Part -d: Use plot function to visualize / show clusters
# plot(kResult)

