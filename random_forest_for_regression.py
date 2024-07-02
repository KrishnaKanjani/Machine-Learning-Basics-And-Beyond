"""Random forest for regression is an ensemble learning algorithm, structurally similar to that for classification but differs in how they evaluate splits and make predictions. Each tree is built using
   a random subset of the training data and a random subset of the features. This randomness is achieved by a technique known as Bootstrap Sampling, which helps to prevent overfitting and makes the 
   model robust against noise. When making predictions, each tree provides an continuous value as output. The final prediction is the average of all the tree outputs instead of the majority voting. As
   the random forest is built on the top of decision trees, we will import the decision tree regressor from our implementation and the rest will be done from scratch, which is mostly similar to Random
   forest classifier. We will use the same Red wine quality dataset from Kaggle, the one we used for decision tree for regression, for this implementation. So, Let's begin!! 
   NOTE: Most of the functions are similar to the ones in Random forest classifier. So, here, I have provided description only for the functions that are different from the Random forest classifier.
   NOTE: The sklearn library is used prepare the training and testing datasets and to compare our results with the sklearn RandomForestRegressor.
"""

# IMPORTING NECESSARY LIBRARIES #
from decision_tree_for_regression import Decision_Tree_Regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# RANDOM FOREST REGRESSOR CLASS #
class Random_Forest_Regressor:
   def __init__(self, no_of_trees=10, max_depth=20, min_samples_split=2, max_features=None):
      self.no_of_trees = no_of_trees
      self.max_depth = max_depth
      self.min_samples_split = min_samples_split
      self.max_features = max_features
      self.trees = []   # list to store the individual decision trees in the forest
   
   def fit(self, x, y):
      self.trees = []
      for _ in range(self.no_of_trees):
         tree = Decision_Tree_Regressor(max_depth = self.max_depth,
                                        min_samples_split = self.min_samples_split,
                                        max_features = self.max_features)
         x_sample, y_sample = self.bootstrap_sampling(x, y)
         tree.fit(x_sample, y_sample)
         self.trees.append(tree)
   
   def bootstrap_sampling(self, x, y):
      no_of_samples = x.shape[0]
      ids = np.random.choice(no_of_samples, no_of_samples, replace=True)
      return x[ids], y[ids]
   
   def make_predictions(self, x_test):
      """Predict the target values for the given data using the trained model
         :param x_test : test input features
         :return : array of predicated target value
      """
      tree_predictions = np.array([tree.make_predictions(x_test) for tree in self.trees])  # predictions from each tree
      final_predictions = np.mean(tree_predictions, axis=0)  #  # for each instance, compute average of predicted values from the all the trees 
      return np.array(final_predictions)
   
   def mse(self, y_test, y_predicted):
      """Determine the Mean Squared Error (MSE) between the true and predicted values.
         :param y_test : true values of the target variable
         :param y_predicted : predicted values of target variable
         :return : MSE of the model
      """
      return np.mean(np.square(y_test - y_predicted))
   
# EXAMPLE USAGE ON THE WINE DATASET #
if __name__ == "__main__":
   data = pd.read_csv("datasets\decision-tree-regression-data.csv")
   x = data.iloc[:, :-1].values
   y = data['quality'].values
   ## Uncomment below 4 lines and comment above 3 lines to check the results on California housing dataset. The results might take around 5-10 
   ## mins, as the dataset is big & I haven't implemented parallel processing here. Also, it depends on the no_of_trees, max_depth values you give.
   # from sklearn import datasets                             
   # data = datasets.fetch_california_housing()
   # x = data.data
   # y = data.target
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
   print("True target values:\n", y_test)

   # Training the random forest model without using sklearn library 
   rfr = Random_Forest_Regressor(no_of_trees=7, max_depth=15)
   rfr.fit(x_train, y_train)

   # Training the Sklearn random forest model
   sklearn_rfr = RandomForestRegressor(n_estimators=7, max_depth=15, random_state=42)
   sklearn_rfr.fit(x_train, y_train)

   # Predictions from both the models
   our_predictions = rfr.make_predictions(x_test)
   print("\nOur Predictions:\n", our_predictions)
   sklearn_prediction = sklearn_rfr.predict(x_test)
   print("\nSklearn Predictions:\n", sklearn_prediction)

   # MSE from both the models
   our_mse = rfr.mse(y_test, our_predictions)
   print("\nMean Squared Error from Our model:", round(our_mse, 5))
   sklearn_mse = mean_squared_error(y_test, sklearn_prediction)
   print("Mean Squared Error from Slearn model:", round(sklearn_mse, 5))