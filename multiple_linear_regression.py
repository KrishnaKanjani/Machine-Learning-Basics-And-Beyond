'''Multiple linear regression is a statistical method used to model the relationship between multiple independent variables (features) and one or more dependent variables 
  (targets). It's employed to understand how changes in independent variables affect the dependent variables. The careful selection of features directly influences 
   predictive accuracy. Through iterative adjustments, the model assigns weights to the features, aiming to align closely with the dataset. For this implementation, 
   We have used the Energy Efficiency Dataset from UCI Machine Learning Repository, in which we will implement multiple linear regression to predict Heating Load and 
   Cooling Load based on various building energy efficiency features. The purpose of this implementation is to try not to use any inbuilt libraries to demonstrate 
   fundamental concepts like gradient descent for optimization and error calculation using a cost function and implement multiple linear regression from scratch.
   NOTE: The sklearn library is used just to compare our results with the in-built Linear Regression function.
   Lets begin !!
'''

# IMPORTING NECESSARY LIBRARIES
from ucimlrepo import fetch_ucirepo 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# GET DATASET
def get_data(): 
    """Get dataset of Energy Efficiency
       The dataset contains features related to building energy efficiency and target variables for 
       heating load and cooling load.
       :return : dataset obtained from the link
    """
    # fetch dataset 
    energy_efficiency = fetch_ucirepo(id=242) 
    X = energy_efficiency.data.features 
    y = energy_efficiency.data.targets 
    dataset = np.column_stack((X, y))  # Combine features and targets into one dataset
    return dataset

# NORMALIZE DATA
def normalize_data(data, scaler=None):
    """Normalize the data using MinMaxScaler. 
       This is done to scale the values of a dataset into a specific range, typically to improve model convergence.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        # Reshape data if it's 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_scaled = scaler.fit_transform(data)
    else:
        # Reshape data if it's 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        data_scaled = scaler.transform(data)
    return data_scaled, scaler

# COST FUNCTION 
def cost_function(x, y, w, len_data):
    """Cost function is the function that finds the error or difference between the predicted and actual values. 
       Cost function formula: J(w) = sum((predicted - actual)^2) / len_data
       :param x    : contains our dataset
       :param y    : contains the output (result vector)
       :param len_data  : length of the dataset
       :param w         : contains the feature vector
       :return          : sum of square error computed from given features
    """
    fx = np.dot(w, x.transpose())        # Predicted
    cost_j = ( (1/ (len_data)) * np.sum(np.square(fx - y.transpose())) )    # Cost function formula
    return cost_j

# GRADIENT DESCENT
def run_gradient_descent(x, y, w, alpha, len_data):
    """Run gradient descent and update the feature vector accordingly.
       Gradient descent is an iterative first-order optimisation algorithm, used to find a local minimum/maximum 
       of a given function. It is used to minimize the error in cost function.
       Formula for updating w: w = w - [ alpha/len_data * sum( (predicted - actual) * x ) ]
       :param x   : contains the dataset
       :param y   : contains the output associated with each data-entry
       :param len_data : length of the data
       :param alpha    : Learning rate of the model
       :param w        : Feature vector (weights for our model features)
       :return         : Updated feature vector using gradient descent
    """
    fx = np.dot(w, x.transpose())  # Predicted
    dw = (1 / len_data) * np.dot((fx - y.transpose()), x).flatten()    # d / dw (cost) i.e derivative of cost function w.r.t dw
    w = w - alpha * dw      # Update w
    return w

# RUN LINEAR REGRESSION FOR MULTIPLE FEATURES AND TARGETS
def run_linear_regression(x, y):
    """Implement linear regression over the dataset for multiple targets (here, 2)"""
    iterations = 100000
    alpha = 0.1  # Learning rate: tells how big or small the steps are taken to reach minimum, should be b/w 0 and 1
    no_features = x.shape[1]
    len_data = x.shape[0]

    np.random.seed(0) 
    w = np.random.randn(2, no_features) * 0.01  # Small random value for each target
    
    # Loop for each target variable
    for i in range(iterations): 
        for j in range(2):  
            w[j] = run_gradient_descent(x, y[:, j], w[j], alpha, len_data)
            error = cost_function(x, y[:, j], w[j], len_data)
            # print(f"At Iteration {i + 1} - Target {j+1} Error is {error:.5f}")      # Uncomment to see the error updations
    return w

# PREDICTION FUNCTION FOR NEW DATA
def make_predictions(new_data, w):
    """Predict the target variables (Heating Load and Cooling Load) for new data"""
    predictions = np.zeros((new_data.shape[0], 2))  # Initialize predictions array for both targets
    
    # Loop for each target variable
    for j in range(2):             
        predictions[:, j] = np.dot(w[j], new_data.transpose())
    return predictions

# MAIN
def main():
    """Driver function"""
    data = get_data()
    len_data = data.shape[0]
    x = np.c_[np.ones(len_data), data[:, :-2]].astype(float)   # Add intercept term (bias) and select all columns except last
    y = data[:, -2:].astype(float)   # Select last 2 columns as targets (Heating Load and Cooling Load) 

    # Normalize the features
    x_normalized, scaler_x = normalize_data(x)   
    
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=5)

    # Training the LR model W/O using sklearn library and getting the model parameters
    w = run_linear_regression(X_train, y_train)
    print("Weight Vector:\n", w)

    # Training the LR model using scikit-learn 
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making predictions on testing data using our implementation
    predictions = make_predictions(X_test, w)
    print("\nOur Predictions:\n", predictions)

    # Making predictions on testing data using sklearn library
    scikit_predictions = model.predict(X_test)
    print("\nSklearn Model Predictions:\n",scikit_predictions)

    # Calculating Mean Squared Error (MSE) for both models
    custom_mse = np.mean(np.square(predictions - y_test))
    print("\nOur Linear Regression MSE:", custom_mse)
    scikit_mse = np.mean(np.square(scikit_predictions - y_test))
    print("Sklearn Linear Regression MSE:", scikit_mse)

if __name__ == "__main__":
    main()