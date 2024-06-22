""" Logistic regression is a machine learning algorithm which is widely used for classification problems. The goal of binary classification is to analyze the relationship 
    between two data classes i.e to predict the probability that an instance belongs to a given class or not. For example, 0/1, true/false, email spam/not spam, tumour 
    malignant or not etc. Instead of giving the exact value as 0 and 1, the model gives the probabilistic values which lies between 0 and 1. For this, we use sigmoid 
    function, that takes input as independent variables and produces a probability value between 0 and 1. In Logistic regression, instead of fitting a regression line, we
    fit an “S” shaped logistic function, which predicts two maximum values (0 or 1). For this implementation, we will use the breast-cancer dataset from Kaggle and predict
    whether the tumour is malignant(1) or benign(0). The purpose of this implementation is to try not to use any built-in ML libraries to demonstrate fundamental concepts like 
    sigmoid function, cost function, logistic loss etc and implement logistic regression from scratch.
    NOTE: The sklearn library is used just to compare our results.
    Lets begin !!
"""

# IMPORTING NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# GET DATA
def get_data():
    """"Get the features and target values from the data"""
    data = pd.read_csv('datasets\logistic-regression-data.csv')
    x = data.iloc[:,:-1]  # Selecting all columns except the last one.
    y = data['diagnosis'].values   # last column as target
    return x ,y

# NORMALIZE DATA
def normalize_data(data, scaler=None):
    """Normalizing the data using StandardScaler. 
       This is done to scale the values of a dataset into a specific range, typically to improve model convergence.
    """
    if scaler is None:
        scaler = StandardScaler()
    # Reshape data if it's 1D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

# SIGMOD FUNCTION
def sigmoid_function(z):
    """Sigmoid function, also known as logistic function is a link function that maps the linear combination of
       input features to a probability. It outputs the value between 0 an 1.
       Formula of sigmoid function is: g(z) = 1 / (1 + e^-z)
       :param z  : input of sigmoid function
       :return   : a value between 0 and 
    """
    gz = 1 / (1 + np.exp(-z))
    return gz
    
# LOGISTIC LOSS
def logistic_loss(y, y_hat):
    """Logistic loss, also known as loss function measures how well a model is doing on a single training datapoint,
       and summing up the losses, we get to know how well the model is doing on the entire dataset. For instance, if the 
       predicted value is closer to the actual value, then the loss is less and hence the model is doing well. Same, vice
       versa.
       Loss function formuls is: loss = -log(y_hat) ;if y=1  AND  loss = -log(1 - y_hat)  ;if y=0
       :param y        : true value of target
       :param y_hat    : predicted value of target
       :return         : loss
    """
    # We will use vectoized version of loss calculation for efficiency
    loss = - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return loss

# COST FUNCTION
def cost_function(x, y, w, b):
    """The cost function for logistic regression is the average of summation of the losses of the data points in a 
       dataset.
       Cost function formula: J(w,b) = 1/m * sum(losses of all individual datapoints) 
       :param x   : dataset feature values
       :param y   : dataset target values
       :param w   : weight assigned to the feature or slope
       :param b   : bias / intercept
       :return    : cost
    """
    m = len(y)
    fx = np.dot(x, w) + b
    y_hat = sigmoid_function(fx)
    loss = logistic_loss(y, y_hat)
    cost_j = (1 / m) * np.sum(loss)
    return cost_j

# GRADIENT DESCENT
def run_gradient_descent(x, y, w, b, alpha):
    """Gradient Descent is an iterative first-order optimisation algorithm, used to find a local minimum/maximum 
       of a given function. It is used to minimize the error in cost function.
       Formula for updating w: w = w - [ alpha/len_data * sum( (predicted - actual) * x )
       Formula for updating b: b = b - [ alpha/len_data * sum( (predicted - actual) )
       :param x   : dataset feature values
       :param y   : dataset target values
       :param w   : weight assigned to the feature or slope
       :param b   : bias / intercept
       :param alpha : learning rate of the model
       :return      : updated values for model prameters w, b
    """
    m = len(y)
    fx = np.dot(w, x.transpose()) + b     
    y_hat = sigmoid_function(fx)
    dw = (1 / m) * np.dot((y_hat - y), x)  # d / dw (cost) i.e derivative of cost function w.r.t dw
    db = (1 / m) * np.sum(y_hat - y)     # d / db (cost) i.e derivative of cost function w.r.t db
    w = w - alpha * dw      # Update w
    b = b - alpha * db      # Update b
    return w, b

# RUN LOGISTIC REGRESSION
def run_logistic_regression(x, y):
    """Implement logistic regression over the dataset"""
    iterations = 100000
    alpha = 0.0001       # Learning rate: tells how big or small the steps are taken to reach minimum, should be b/w 0 and 1
    no_features = x.shape[1]
    
    np.random.seed(0) 
    w = np.random.randn(no_features) * 0.01  # Initialize small random value for the target
    b = -0.01    # Initialize bias

    for i in range(iterations):
        w, b = run_gradient_descent(x, y, w, b, alpha)
        error = cost_function(x, y, w, b)
        # print(f"At Iteration {i + 1}, Error is {error:.5f}")    # Uncomment to see the error updations
    return w, b

# PREDICT FUNCTION
def make_predictions(new_data, w, b):
    """Predict the class labels for a given input using learned model parameters."""
    fx = np.dot(new_data, w) + b
    y_hat = sigmoid_function(fx)
    return (y_hat >= 0.5).astype(int)

# MAIN FUNCTION
def main():
    x, y = get_data()
    # Normalize the features
    x_normalized, scaler_x = normalize_data(x)  

    # Split data into training and testing sets (80% training, 20% testing)
    x_train , x_test ,y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=2)

    # Training the LR model W/O using sklearn library and getting the model parameters
    w, b = run_logistic_regression(x_train, y_train)  
    print(f"\nModel Parameters W/O using sklearn Library\nw: {w}\nb: {b}")

    # Training the LR model using scikit-learn 
    model = LogisticRegression()
    model.fit(x_train, y_train)
    print(f"\nModel Parameters using sklearn Library\nw: {model.coef_}\nb: {model.intercept_}")

    # Making predictions on testing data using our implementation
    predictions = make_predictions(x_test, w, b)
    print("\nOur Predictions:\n", predictions)
    
    # Making predictions on testing data using sklearn library
    scikit_predictions = model.predict(x_test)
    print("\nSklearn Model Predictions:\n",scikit_predictions)

    # Calculating accuracy of both the models
    our_accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy of our model: {our_accuracy * 100:.2f}%")
    scikit_accuracy = accuracy_score(y_test, scikit_predictions)
    print(f"Accuracy of Sklearn Model: {scikit_accuracy * 100:.2f}%")
    
if __name__ == "__main__":
    main()

