library(randomForest)
library(naniar)
library(coop)
library(tm)
library(tidyverse)
library(tidytext)
library(caret)
library(pROC)
library(prediction)
library(mlr)
library(doParallel)
library(grid)
library(DMwR)
library(Hmisc)
library(mlr)

sparsedoc_word <- read.csv(file='/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week12/sparsedoc_word.csv', header=FALSE, sep=",")
labels <- read.delim(file='/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week12/fraudlabels.txt', header=FALSE)

#sum(is.na(sparsedoc_word$X47))
#visdat::vis_miss(labels)
sparsedoc_curated<-sparsedoc_word[sparsedoc_word$V2%in%c(1:5000),]

(docs_dtm <- sparsedoc_curated %>%
    # get count of each token in each document
    
    # create a document-term matrix with all features and tf weighting
    cast_dtm(document = V1, term = V2, value = V3))

removeSparseTerms(docs_dtm, sparse = .99)

combined_df <- as.data.frame(cbind(as.data.frame(as.matrix(docs_dtm)),labels))
new_df <- combined_df
new_df$V1 <- as.factor(final_df$V1)

# Random sampling 80/20 train test split
sample_size <- floor(0.80 * nrow(new_df))
set.seed(123)
train_index <- sample(seq_len(nrow(new_df)), size = sample_size)
train_data <- new_df[train_index,]
test_data <- new_df[-train_index,]
# Random forest does not like the given integer names so I added a prefix to the column names
colnames(train_data) <- paste("Term", colnames(train_data), sep= "_")
colnames(test_data) <- paste("Term", colnames(test_data), sep= "_")

# Train rf model without mtry
forest.model <- randomForest(train_data$Term_V1 ~ ., data=train_data, ntree=100) # , keep.forest=TRUE, importance=TRUE
# Trial 1 with 100 decision trees, no tuning
predicted <- predict(forest.model, newdata = test_data, type = "prob")[,2]
cm_pred <- predict(forest.model, newdata = test_data, type="response")
# Trial 1 AUC = 0.94
roc_obj <- roc(test_data$Term_V1, predicted)
auc(roc_obj)

# Trial 1 Accuracy = 0.87
CM = table(predicted, test_data$Term_V1)
accuracy = (sum(diag(CM)))/sum(CM)
accuracy

# Trial 1 plot and importance
plot(forest.model)
importance(forest.model)

# Trial 2 with best mtry, optimal mtry = 105
bestmtry <- tuneRF(train_data[-5001], train_data$Term_V1, ntreeTry=100,
                   stepFactor = 1.5, improve= 0.01, trace = TRUE, plot = TRUE,
                   dobest = FALSE)

# Trial 2 - Train model with 105 mtry and parameters and 500 d-trees
forest.model2 <- randomForest(train_data$Term_V1 ~ ., data=train_data, 
                              mtry = 105, ntree=500, keep.forest=TRUE,
                             importance=TRUE, test = test_data$Term_V1) # , keep.forest=TRUE, importance=TRUE
plot(forest.model2)
importance(forest.model2)
varImpPlot(forest.model2)
predicted2 <- predict(forest.model2, newdata=test_data, type="prob")[,2]
cm_pred2 <- predict(forest.model2, newdata=test_data, type = "response")

# Trial 2 AUC = 0.95
roc_obj <- roc(test_data$Term_V1, predicted2)
auc(roc_obj)

# Trial 2 Accuracy = 0.88
CM2 = table(cm_pred2, test_data$Term_V1)
accuracy = (sum(diag(CM2)))/sum(CM2)
accuracy


# WEEK 12 Random Forest

new_dataset <- read.csv(file='/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week12/gopi_final_dtm_latest.csv', header=TRUE, sep=",", fileEncoding='latin1')
head(new_dataset,n=10)
#randomly shuffle data
shuffled_data <- new_dataset[sample(nrow(new_dataset)),]

# 10 fold cross validation
#Create 10 equal folds
folds <- cut(seq(1,nrow(shuffled_data)), breaks = 10, labels = FALSE)
shuffled_data$holdoutpred <- rep(0,nrow(shuffled_data))
for(i in 1:10) {
  testIndexes <- which(folds == i, arr.ind = TRUE)
  testData <- shuffled_data[testIndexes, ]
  trainData <- shuffled_data[-testIndexes, ]
  
}

shuffled_data$holdoutpred


# New way

#train_control <- trainControl(method="cv", number = 10)
set.seed(101)
new_dataset <- read.csv(file='/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week12/gopi_final_dtm_latest.csv', header=TRUE, sep=",", fileEncoding='latin1')
new_dataset$doc_name <- NULL
new_dataset$company_name <- NULL
#new_dataset <- sapply(new_dataset[4,],as.numeric)
new_dataset$year <- as.factor(new_dataset$year)
new_dataset$label <- as.factor(new_dataset$label)
shuffled_data <- new_dataset[sample(nrow(new_dataset)),]
sample_size <- floor(0.80 * nrow(shuffled_data))
train_index3 <- sample(seq_len(nrow(shuffled_data)), size = sample_size)
train_data3 <- shuffled_data[train_index3,]
test_data3 <- shuffled_data[-train_index3,]

new.rf.model <- randomForest(train_data3$label~., data=train_data3,
                            ntree=100, keep.forest=TRUE,
                             importance=TRUE, test = test_data$label)

View(head(new_dataset,n = 100))

plot(new.rf.model, main="forest.model3")
importance(new.rf.model)
varImpPlot(new.rf.model, main="")
predicted3 <- predict(new.rf.model, newdata=test_data3, type="prob")[,2]
cm_pred3 <- predict(new.rf.model, newdata=test_data3, type = "response")

roc_obj <- roc(test_data3$label, predicted3)
auc(roc_obj)

CM = table(cm_pred3, test_data3$label)
accuracy = (sum(diag(CM)))/sum(CM)
accuracy




CM2 = table(cm_pred2, test_data$Term_V1)
accuracy = (sum(diag(CM2)))/sum(CM2)
accuracy

plot(as.integer(new_dataset$year),new_dataset$label)
boxplot(new_dataset$label~new_dataset$year, data=new_dataset)
plot(new_dataset$risk_level, new_dataset$label)


# Balanced Random Forest - bringing up lower classes 
#train_control <- trainControl(method="cv", number = 10)

new_dataset <- read.csv(file='/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week12/gopi_final_dtm_latest.csv', header=TRUE, sep=",", fileEncoding='latin1')
new_dataset$doc_name <- NULL
new_dataset$company_name <- NULL
new_dataset$year <- as.factor(new_dataset$year)
new_dataset$label <- as.factor(new_dataset$label)
#new_dataset <- sapply(new_dataset[4,],as.numeric)

# corr analysis
df_corrTest = apply(new_dataset, 2, function(x) as.numeric(as.character(x)));
plot(varclus(df_corrTest, similarity="spearman"))
abline (h = 1 - 0.8 , col=" grey ", lty =2)

# Check imbalance
# rows: 4570 False (0) and 1127 True (1)
length(which(new_dataset$label == '0'));
length(which(new_dataset$label == '1'));

# Train test split 80/20
set.seed(101)
shuffled_data <- new_dataset[sample(nrow(new_dataset)),]
sample_size <- floor(0.80 * nrow(shuffled_data))
train_index4 <- sample(seq_len(nrow(shuffled_data)), size = sample_size)
train_data4 <- shuffled_data[train_index4,]
test_data4 <- shuffled_data[-train_index4,]

balanced.data <- SMOTE(label~.,train_data4, perc.over = 300, k=5)

#as.data.frame(table(balanced.data$label))

#bestmtry <- tuneRF(balanced.data[-5006], balanced.data$label, ntreeTry=100,
                 #  stepFactor = 1.5, improve= 0.01, trace = TRUE, plot = TRUE,
                #   dobest = FALSE)

set.seed(123)
shuffled_data <- new_dataset[sample(nrow(new_dataset)),] # shuffling data
folds <- cut(seq(1,nrow(shuffled_data)),breaks=10,labels=FALSE) #10 fold cross validation

aucV = c()
fMeasV = c()
accuV = c()

for(i in 1:10){
  ind <- (((i-1) * round((1/10)*nrow(shuffled_data))) + 1):((i*round((1/10) * nrow(shuffled_data))))
  # Exclude them from train set
  train <- shuffled_data[-ind ,]
  # Include them in test set
  test <- shuffled_data[ind ,]
  
  train <- SMOTE(label ~ ., train, perc.over = 300)#,perc.under=100)
  
  as.data.frame(table(train$label))
  
  rf.model <- randomForest(label~.,
                      data=train,
                      ntree = 25,
                      mtry = 5,
                      importance =TRUE,
                      proximity = TRUE)
  #Trial 1 fm = 0.92
  predpr <- predict(rf.model, newdata=test, type="prob")[,2]
  cm_pred <- predict(rf.model, newdata=test, type="response")
  fm <- F_meas(cm_pred, test$label)
  fMeasV <- union(fMeasV, c(fm)) 
  
  #Trial 1 accuracy = 0.86
  cm4 = table(cm_pred,test$label)
  accuracy = (sum(diag(cm4)))/sum(cm4)
  accuV <- union(aucV,accuracy)
  
  #Trial 1 auc = 0.91
  roc_obj <- roc(test$label, predpr)
  auc = auc(roc_obj)
  aucV <- union(aucV, c(auc))
  
  plot(rf.model)
  varImpPlot(rf.model)
}
save(rf.model,file="/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week13/rf_model_10_fold.rda")
load("/Users/admin/Documents/Queens_Masters_Courses/CISC873/Assignment_RF/Dataset\ 3/Week13/rf_model_10_fold.rda")

# Plotting
library(ggplot2)
library(reshape2)
a <- data.frame(F_Measure = fMeasV)
b <- data.frame(Accuracy = accuV)
c <- data.frame(Area_Under_Curve = aucV)
myList <- list(a,b,c)
df_box <- melt(myList)
#Separate boxplots for each data.frame
qplot(factor(variable), value, data = df_box, geom = "boxplot",
      xlab = '10 fold Cross Validation Evaluation Metrics',
      ylab = 'Value')

#-------------- TUNE TRAIN + rf plots -----------------

rf1 <- randomForest(label~.,data=train,
                    ntree = 25,
                    
                    importance =TRUE,
                    proximity = TRUE)

tuneTrain = subset(df, select=-c(fraud))
t <- tuneRF(tuneTrain, df$fraud, 
            stepFactor = 0.8,
            plot=TRUE,
            ntreeTry = 25,
            trace = TRUE,
            improve = 0.1)

#"Number of nodes of tree"
hist(treesize(rf.model),
     main = "Number of tree nodes",
     col = 'blue',xlab="Size of tree")
plot(rf.model)
varImpPlot(rf.model)
importance(rf.model)#same as above-shows values
varUsed(rf.model)

#Partical dependence of a particular feature on the output
partialPlot(rf.model, tuneTrain, df$fraud, "1")

#toget info about a specific tree
getTree(rf.model, 1, labelVar = TRUE)

#multi-dimensional scaling of proximity matrix 
MDSplot(rf.model, new_dataset$label)
