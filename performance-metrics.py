"""Performance metrics are the calculations used to evaluate the performance or quality of the model. These performance metrics help us understand how well our model has
   performed for the given data. In this way, we can improve the model's performance by tuning the hyper-parameters. In machine learning, mostly tasks or problems are
   divided into classification and regression. Not all metrics can be used for all types of problems; hence, it is important to know and understand which metrics should
   be used. Different evaluation metrics are used for both Regression and Classification tasks. So, in this file, I have implemented performance metrics from scratch that
   are used for classification and regression tasks.
"""
import numpy as np

#-------------------------------------------------------------------- PERFORMANCE METRICS FOR REGRESSION ------------------------------------------------------------------#
"""Regression is a supervised learning technique that aims to find the relationships between the dependent and independent variables. A predictive regression model predicts 
  a numeric or discrete value. So, the performance of a regression model is reported as errors in the prediction. Following are the popular metrics that are used to evaluate 
  the performance of Regression models:
"""

# 1. Mean Absolute Error (MAE)
def mae(y, y_predicted):
    """Mean Absolute Error or MAE is one of the simplest metrics, which measures the absolute difference between actual and
       predicted values, where absolute means taking a number as positive.
       Formula: mae = 1/n * sum(abs(y - y_predicted))
    """
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    diff = abs(y - y_predicted)
    mae = np.mean(diff)
    return mae

# 2. Mean Squared Error (MSE)
def mse(y, y_predicted):
    """Mean Squared error or MSE is one of the most suitable metrics for regression evaluation. It measures the average of the
       squared difference between predicted values and the actual value given by the model.
       Formula: mse = 1/n * sum((y - y_predicted) ^2))
    """
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    squared_diff = np.square(y - y_predicted)
    mse = np.mean(squared_diff)
    return mse 

# 3 Root Mean Squared Error (RMSE)
def rmse(y, y_predicted):
    """Root Mean Squared Error or RMSE measures the average difference between values predicted by a model and the actual values. It 
       provides an estimation of how well the model is able to predict the target value (accuracy). The lower the value of the Root 
       Mean Squared Error, the better the model is.
       Formula: rmse = sqrt(mse)
    """
    y = np.array(y)
    y_predicted = np.array(y_predicted)
    squared_diff = np.square(y - y_predicted)
    mse = np.mean(squared_diff)
    # mse_ = mse(y, y_predicted)
    rmse = np.sqrt(mse)
    return rmse

# 4. R Squared Score 
def r2score(y, y_predicted):
    """The coefficient of determination also called the R2 score is used to evaluate the performance of a linear regression model.
       It is used to check how well-observed results are reproduced by the model, depending on the ratio of total deviation of results 
       described by the model. The R2 score will always be less than or equal to 1 without concerning if the values are too large or small.
       Formula: r2 = 1 - SSE/SST, where 
        SSE: Sum of Squared Errors, represents the variability in the dependent variable (y) that is not explained by your fitted model. It 
            measures how much the predicted values(y_predicted) from your model deviate from the actual values(y).
        SST: Sum of Squares Total, represents the total variability in the dependent variable (y) around its mean. It measures how spread 
            out the actual data points (y values) are from the average value (y mean).
    """
    # Calculating SSE
    sse = np.sum( np.square(y - y_predicted) )
    # Calculating SST
    y_mean = np.mean(y)
    sst = np.sum( np.square(y - y_mean) )
    # Calculatin R2
    r2 = 1 - (sse/sst)
    return r2

# 5. Adjusted R Squared Score 
def adjusted_r2(y, y_predicted, n, k):
    """Adjusted R-squared, is a refinement of the standard R-squared (R²) metric. While R² measures the proportion of variance in 
       the dependent variable explained by the model, it has a tendency to overestimate the model's performance, especially when you 
       add more independent variables. So to overcome the issue, adjust r2 is used, which adjusts the values of increasing independent 
       variables and only shows improvement if there is a real improvement.
       Here, n: no. of samples(data points) AND k: no. of independent variables
       Formula: adjusted_r2 = 1 - [ (1-R2) * ( (n-1) / (n-k-1) ) ]
    """
    sse = np.sum( np.square(y - y_predicted) )
    y_mean = np.mean(y)
    sst = np.sum( np.square(y - y_mean) )
    r2 = 1 - (sse/sst)
    # r2 = r2score(y, y_predicted)
    adjusted_r2 = 1 - ( (1-r2) * ( (n-1) / (n-k-1) ) )
    return adjusted_r2

#----------------------------------------------------------------- PERFORMANCE METRICS FOR CLASSIFICATION -----------------------------------------------------------------#
"""In a classification problem, the category or classes of data is identified based on training data. The model learns from the given dataset and then classifies the new 
   data into classes or groups based on the training. It predicts class labels as the output. To evaluate the performance of a classification model, different metrics are 
   used. Below are the mostly widely used ones:
"""

# 1. Confusion Matrix
def confusion_matrix(y, y_predicted):
    """A confusion matrix is a tabular representation of prediction outcomes of any classifier, which is used to describe the performance 
       of the classification model on a set of test data when true values are known. It displays the number of accurate and inaccurate instances
       based on the model’s predictions. It returns, 
        True positives (TP): occurs when the model accurately predicts a positive data point.
        True negatives (TN): occurs when the model accurately predicts a negative data point.
        False positives (FP): occurs when the model predicts a positive data point incorrectly.
        False negatives (FN): occurs when the model mispredicts a negative data point.
        We will also use these values for calculating other performance metrics.
    """
    TP = np.sum( (y == 1) & (y_predicted == 1) )
    TN = np.sum( (y == 0) & (y_predicted == 0) )
    FP = np.sum( (y == 0) & (y_predicted == 1) )
    FN = np.sum( (y == 1) & (y_predicted == 0) )
    return TP, TN, FP, FN

# 2. Accuracy
def accuracy(y, y_predicted):
    """The accuracy metric is one of the simplest classification metrics to implement, and it can be determined as the number of correct
       predictions to the total number of predictions. 
       Formula: accuracy = Total true predictions / Total Predictions
    """
    TP, TN, FP, FN = confusion_matrix(y, y_predicted)
    total_true_predictions = TP + TN
    total_predictions = TP + TN + FP + FN
    accuracy = total_true_predictions / total_predictions
    return accuracy

# 3. Precision
def precision(y, y_predicted):
    """ The precision determines the proportion of positive prediction that was actually correct. It can be calculated as the True Positive 
        or predictions that are actually true to the total positive predictions (True Positive and False Positive).
        Formula: precision = True positives / (True positives + False Positives)
    """
    TP, TN, FP, FN = confusion_matrix(y, y_predicted)
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
        return precision
    else:
        return 0
    
# 4. Recall or Sensitivity
def recall(y, y_predicted):
    """Recall is similar to the Precision metric; however, it aims to calculate the proportion of actual positives that were identified 
       incorrectly, i.e, True Positives to the total number of positives, either correctly predicted as positive or incorrectly predicted 
       as negative (true Positives and false negatives).
       Formula: recall = True positives / (True positives + False Negatives)
    """
    TP, TN, FP, FN = confusion_matrix(y, y_predicted)
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
        return recall
    else:
        return 0

# 5. F1-score
def f1score(y, y_predicted):
    """F1 Score is a metric to evaluate a binary classification model on the basis of predictions that are made for the positive class. It is 
       calculated with the help of Precision and Recall. F1 Score is the harmonic mean of both precision and recall, assigning equal weight to 
       each of them.
       Formula: f1score = 2 * ( (precision * recall) / (precision + recall) )
    """
    p = precision(y, y_predicted)
    r = recall(y, y_predicted)
    f1score = 2 * ( (p * r) / (p + r) )
    return f1score

## EXAMPLE USAGE
# For regression metrics
y_true_regression = np.array([3, -0.5, 2, 7, 10.64, 24, 33.35, -8, 55, 26, 17])
y_pred_regression = np.array([2.5, 0.0, 1.7, 7.01, 10.46, 20, 32.33, -6.9, 57, 26, 17])

print("REGRESSION METRICS:")
print(f"MAE: {round(mae(y_true_regression, y_pred_regression), 4)}")
print(f"MSE: {round(mse(y_true_regression, y_pred_regression), 4)}")
print(f"RMSE: {round(rmse(y_true_regression, y_pred_regression), 4)}")
print(f"R-squared: {round(r2score(y_true_regression, y_pred_regression), 4)}")
print(f"Adjusted R-squared: {round(adjusted_r2(y_true_regression, y_pred_regression, len(y_true_regression), 8), 4)}")

# For classification metrics
y_true_classification = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
y_pred_classification = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1])

print("\nCLASSIFICATION METRICS:")
print(f"Accuracy: {round(accuracy(y_true_classification, y_pred_classification), 4)}")
print(f"Precision: {round(precision(y_true_classification, y_pred_classification), 2)}")
print(f"Recall: {round(recall(y_true_classification, y_pred_classification), 4)}")
print(f"F1-score: {round(f1score(y_true_classification, y_pred_classification), 4)}")    