#import require library
library(AUC)
library(onehot)
library(xgboost)

#normalize Function
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

for (index in 1:3) {

    X_train <- read.csv(sprintf("hw07_target%d_training_data.csv", index), header = TRUE)
    X_train_label <- read.csv(sprintf("hw07_target%d_training_label.csv", index), header = TRUE)
    X_test  <- read.csv(sprintf("hw07_target%d_test_data.csv", index), header = TRUE)
    
    #removes the first variable(id) from the data set.
    X_train <- X_train[,-1]
    
    #removes the variable "TARGET" from the data set.
    X_train_label <- X_train_label[,"TARGET"]
    
    encoder <- onehot(X_train, addNA = TRUE, max_levels = Inf)
    
    #X_train_prediction generate
    X_train_prediction = predict(encoder,X_train)
    
    #X_test_prediction generate
    X_test_prediction = predict(encoder,X_test[,-1])
    
    xgboost_model <- xgboost(
                        data  = X_train_prediction,
                        label = X_train_label,
                        nrounds = 10,
                        objective= "binary:logitraw"
                     )
    training_scores <- predict(xgboost_model, X_train_prediction)
    
    # AUC score for training data
    print(auc(roc(predictions = training_scores, labels = as.factor(X_train_label))))
    
    #Test scoring calculator
    test_scores <- predict(xgboost_model, X_test_prediction)
    
    #Test scoring convert normalize
    test_scores_normalize <- normalize(test_scores)
    
    #Result writing a file
    write.table(
                cbind(ID = X_test[,"ID"], 
                TARGET = test_scores_normalize), 
                file = sprintf("hw07_target%d_test_predictions.csv",index), 
                row.names = FALSE, 
                sep = ",") 
}
