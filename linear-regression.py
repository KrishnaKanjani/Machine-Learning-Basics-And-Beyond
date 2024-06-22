"""Linear Regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. It minimizes the cost
   function, typically the mean squared error or the mean absolute error, by adjusting parameters (slope/weight and intercept/bias) through gradient descent. Predictions 
   are made based on these optimized parameters. For this implementation, we have used the Placement dataset from Kaggle, in which we will implement simple linear regression 
   to predict the Package (target) a student could get based on its CGPA (feature). The purpose of this implementation is to try not to use any inbuilt libraries to 
   demonstrate fundamental concepts like gradient descent for optimization and error calculation using a cost function and implement linear regression from scratch.
   NOTE: The sklearn library is used just to compare our results with the in-built Linear Regression function.
   Lets begin !!
"""

# IMPORTING NECESSARY LIBRARIES
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# GET DATA
def get_data():
    """"Get the features and target values from the data"""
    data = pd.read_csv('datasets\linear-regression-data.csv')
    x = data['cgpa'].values.reshape(-1, 1)
    y = data['package'].values
    return x,y

# COST FUNCTION
def cost_function(x, y, w, b):
    """Cost function is the function that finds the error or difference between the predicted and actual values.
       Cost function formula: J(w,b) = sum((predicted - actual)^2) / len_data
       :param x   : dataset feature values
       :param y   : dataset target values
       :param w   : weight assigned to the feature or slope
       :param b   : bias
       :return    : cost 
    """
    m = len(y)
    fx = np.dot(w, x.transpose()) + b       # Predicted
    cost_j = ( (1/m) * np.sum(np.square(fx - y)))   # Cost function formula
    return cost_j

# GRADIENT DESCENT ALGORITHM TO MINIMIZE J (COST)
def run_gradient_descent(x, y, w, b, alpha):
    """Gradient Descent an iterative first-order optimisation algorithm, used to find a local minimum/maximum 
       of a given function. It is used to minimize the error in cost function.
       Formula for updating w: w = w - [ alpha/len_data * sum( (predicted - actual) * x )
       Formula for updating b: b = b - [ alpha/len_data * sum( (predicted - actual) )
       :param x   : dataset feature values
       :param y   : dataset target values
       :param w   : weight assigned to the feature or slope
       :param b   : bias
       :param alpha : learning rate of the model
       :return      : updated values for model prameters w, b
    """
    m = len(y)
    fx = np.dot(w, x.transpose()) + b     # Predicted
    dw = (1 / m) * np.dot((fx - y), x).flatten()   # d / dw (cost) i.e derivative of cost function w.r.t dw
    db = (1 / m) * np.sum(fx - y)     # d / db (cost) i.e derivative of cost function w.r.t db
    w = w - alpha * dw      # Update w
    b = b - alpha * db      # Update b
    return w, b

# LINEAR REGRESSION IMPLEMENTATION
def run_linear_regression(x, y):
    """Implement linear regression over the dataset"""
    iterations = 100000
    alpha = 0.0013     # Learning rate: tells how big or small the steps are taken to reach minimum, should be b/w 0 and 1
    no_features = x.shape[1]
    
    np.random.seed(0) 
    w = np.random.randn(no_features) * 0.01  # Initialize small random value for the target
    b = 1    # Initialize bias

    # Loop for computing gradient descent and cost function for each iterations
    for i in range(iterations):
        w, b = run_gradient_descent(x, y, w, b, alpha)
        error = cost_function(x, y, w, b)
        # print(f"At Iteration {i + 1}, Error is {error:.5f}")   # Uncomment to see the error updations
    return w, b

# PREDICTION FUNCTION FOR NEW DATA
def make_predictions(new_data, w, b):
    """Predict the target variable (package) for new data"""
    prediction = np.dot(new_data, w) + b
    return prediction

# Main Function
def main():
    """"Driver Function"""
    x, y = get_data()

    # Split data into training and testing sets (80% training, 20% testing)
    x_train , x_test ,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    # Training the LR model w/o using libraries i.e our implementation and getting the model parameters
    w, b = run_linear_regression(x_train, y_train)  
    print(f"\nModel Parameters W/O using ML Libraries\nw: {w}\nb: {b}")

    # Training the LR model using sklearn library and getting the model parameters
    model = LinearRegression()
    model.fit(x_train,y_train)
    wLR = model.coef_
    bLR = model.intercept_
    print(f"\nModel Parameters Using ML Libraries:\nw: {wLR}\nb: {bLR}")

    # Making predictions on testing data using both models
    predictions = make_predictions(x_test, w, b)
    print("\nOur Predictions:\n", predictions)
    scikit_predictions = model.predict(x_test)
    print("\nSklearn Model Predictions:\n", scikit_predictions)

    # Calculating Mean Squared Error (MSE) for both models
    custom_mse = np.mean(np.square(predictions - y_test))
    print("\nOur Linear Regression MSE:", custom_mse)
    scikit_mse = np.mean(np.square(scikit_predictions - y_test))
    print("Sklearn Linear Regression MSE:", scikit_mse)

if __name__ == "__main__":
    main()

