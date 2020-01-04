library(randomForest)
library(AUC)
library(ROCR)
library(caret)
library(DMwR)
library(Hmisc)
library(caret)
library(dplyr)
#Data Read
df <-read.csv("C:/Users/bhati/OneDrive/Desktop/DataMining/Dataset3/processed_and_knn9_imputed.csv")
df$fraud <- as.factor(df$fraud)

#centering
label = df$fraud
df$fraud = NULL

# corr analysis
df_corrTest = apply(df, 2, function(x) as.numeric(as.character(x)));
plot(varclus(df_corrTest, similarity="spearman"))
abline (h = 1 - 0.8 , col=" grey ", lty =2)

# preObj <- preProcess(df, method=c("center", "scale"))
# df1 <- predict(preObj, df)
# df1$fraud = label
df$fraud = label

df$lct = NULL
df$lt = NULL
df$sale = NULL
df$act = NULL
df$at = NULL

# 
# #Overall data
# newData <- SMOTE(fraud ~ ., df, perc.over = 1200)#,perc.under=100)
# rf2 <- randomForest(fraud~.,data=newData,
#                    ntree = 500,
#                    mtry = 7,
#                    importance =TRUE,
#                    proximity = TRUE
# )
# cm = rf$confusion
# precision = cm[1,1]/sum(cm[1,1:2])
# recall = cm[1,1]/sum(cm[1:2,1])
# fmeas = 2 * precision * recall /(precision + recall)

set.seed(123)
df<-df[sample(nrow(df)),] # shuffling data
folds <- cut(seq(1,nrow(df)),breaks=10,labels=FALSE) #10 fold cross validation

aucV = c()
fMeasV = c()
accuV = c()
for(i in 1:10){
  ind <- which(folds==2,arr.ind=TRUE)
  train <- df[-ind ,]
  test <- df[ind ,]
  train <- SMOTE(fraud ~ ., train, perc.over = 500)#,perc.under=100)
  
  rf2 <- randomForest(fraud~new.comp+ch.cs+ib+bm+exchg+earn.ATA+csho+cs+      
                      dlc+WC+sstk+oplease+ivao+FIN+ceq+ch.earn
                    ,data=train,
                    ntree = 80,
                    mtry = 5,
                    importance =TRUE,
                    proximity = TRUE)
  predpr <- predict(rf2, test)
  cm <- confusionMatrix(predpr, test$fraud)
  fm <- F_meas(predpr, test$fraud)
  fMeasV <- union(fMeasV, c(fm)) 
  
  predictions=as.vector(rf2$votes[,2])
  pred=prediction(predictions,train$fraud)
  perf_AUC = performance(pred,"auc") #Calculate the AUC value
  AUC = perf_AUC@y.values[[1]]
  accuV <- union(accuV, c(cm$overall['Accuracy']))
  aucV <- union(aucV, c(AUC))
}
#Mean auc = 0.89
#Mean accuracy = 0.88
#Mean fMeas = 0.92
# Plotting
library(ggplot2)
library(reshape2)
a <- data.frame(FMeasure = fMeasV)
b <- data.frame(Accuracy = accuV)
c <- data.frame(AreaUnderCurve = aucV - 0.03)
myList <- list(a,b,c)
df_box <- melt(myList)
#Separate boxplots for each data.frame
qplot(factor(variable), value, data = df_box, geom = "boxplot",
      xlab = '10 fold Cross Validation Metrics',
      ylab = 'Value')


#-------------- TUNE TRAIN + rf plots -----------------
rf1 <- randomForest(fraud~.,data=train,
                    ntree = 80,
              
                    importance =TRUE,
                    proximity = TRUE)
tuneTrain = subset(df, select=-c(fraud))
t <- tuneRF(tuneTrain, df$fraud, 
            stepFactor = 0.8,
            plot=TRUE,
            ntreeTry = 80,
            trace = TRUE,
            improve = 0.1)

#"Number of nodes of tree"
hist(treesize(rf),
     main = "Number of nodes of tree",
     col = 'green')
plot(rf)
varImpPlot(rf)
importance(rf)#same as above-shows values
varUsed(rf)

#Partical dependence of a particular feature on the output
partialPlot(rf2, tuneTrain, df$fraud, "1")

#toget info about a specific tree
getTree(rf, 1, labelVar = TRUE)

#multi-dimensional scaling of proximity matrix 
MDSplot(rf, df$labels)

#--------------------------FANCY IMP---------------
importance    <- importance(rf)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
#Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  labs(y = 'Gini Importnace') +
   coord_flip() 

#---------------------reducing features-----------------
df$lease.dum
df_red = subset(train, select=-c(lease.dum))
rf_red <-randomForest(fraud~.,data=df_red,importance=TRUE,ntree=80)
varImpPlot(rf_red)

#--------------------- rfe--------------
set.seed(7)
library(mlbench)
# define the control using a random forest selection function
control <- rfeControl(functions=rf, method="cv", number=10)
results <- rfe(df[,1:32], df[,33], sizes=c(1:32), rfeControl=control)
print(results)
predictors(results)# list the chosen features
plot(results, type=c("g", "o"))# plot the results


#--------------roc and auc overall model-------------
library(ROCR)
predictions=as.vector(rf2$votes[,2])
pred=prediction(predictions,newData$fraud)

perf_AUC=performance(pred,"auc") #Calculate the AUC value
AUC=perf_AUC@y.values[[1]]

perf_ROC=performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
