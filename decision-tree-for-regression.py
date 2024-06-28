"""Decision tree for Regression is structurally similar to that for classification but differs in how they evaluate splits and make predictions. At each node, the decision tree regressor selects 
   the feature and split point that minimize the sum of squared residuals between the predicted and actual values of the target variable. For choosing the best feature to spilt, concepts like
   Variance reduction or Mean squared error reduction are used for regression. This process continues until a stopping criteria is met, such as reaching a maximum tree depth or no further improvement
   in the residual reduction. Each leaf node represents the predicted value for regression tasks. Decision trees for regression are effective for capturing non-linear relationships in data but can 
   overfit, which can be mitigated using pruning techniques. In this implementation, I'll use Pre-Pruning and to demonstrate all these concepts, I have implemented the algorithm from scratch. I have 
   used the Red wine quality dataset from Kaggle for this implementation. So, Let's begin !! 
   NOTE: Most of the functions are similar to the ones in Decison tree classifier. So, here, I have provided description only for the functions that are different from the Decison tree classifier.
   NOTE: The sklearn library is used just to prepare the training and testing datasets and to compare our results with the sklearn DecisionTreeRegressor.
"""

# IMPORTING NECESSARY LIBRARIES #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# NODE CLASS #
"""The Node class will represent each node in the decision tree. It will store the information about: (1) the 'feature' to split the tree on; (2) the 'threshold' value for splitting; 
   (3) the 'left' and 'right' child nodes of each node; (4) 'value' which is a predicted target value if the node is a leaf node. The default value for all will be None.
"""
class Node:
   def __init__(self, feature=None, threshold=None, left=None, right=None, *,value=None ):
      self.feature = feature
      self.threshold = threshold
      self.left = left
      self.right = right
      self.value = value

# DECISION TREE REGRESSOR #
"""The Decision_Tree_Regressor class is responsible for doing the following: (1) Handle the maximum depth, minimum samples and max features for splits (Pre-pruning); (2) Compute variance
   reduction to evaluate splits; (3) Identify the best feature and threshold for splitting the data (Best Split using Variance reduction); (4) Recursively grow the tree by finding the best
   split; (5) Predict values for input data by traversing the tree.
"""
class Decision_Tree_Regressor:
   def __init__(self, max_depth=None, min_samples_split=None, max_features=None):
      self.max_depth = max_depth
      self.min_samples_split = min_samples_split
      self.max_features = max_features
      self.root_node = None

   def fit(self, x, y):
      if not self.max_features:   
         self.max_features = x.shape[1] 
      else:
         self.max_features = min(x.shape[1], self.max_features)

      self.root_node = self.grow_tree(x, y)

   def grow_tree(self, x, y, depth=0):
      leaf_value = self.calculate_leaf_value(y)
      node = Node(value = leaf_value)

      # Check the stopping criteria for recursion
      if self.max_depth is not None and depth >= self.max_depth or x.shape[0] < self.min_samples_split:
         return node
      
      else:
         best_feature, best_threshold = self.best_split(x, y)
         if best_feature is None or best_threshold is None:
            return node
         else:
            left_child_node = np.where(x[:, best_feature] <= best_threshold)[0]
            right_child_node = np.where(x[:, best_feature] > best_threshold)[0]

            if len(left_child_node) == 0 or len(right_child_node) == 0:
               return node
            
            left = self.grow_tree( x[left_child_node, :], y[left_child_node], depth + 1 )
            right = self.grow_tree( x[right_child_node, :], y[right_child_node], depth + 1 )
            return Node(best_feature, best_threshold, left, right)

   def best_split(self, x, y):
      """Find the best feature and threshold for splitting the data, using variance reduction.
         :param x : feature matrix
         :param y : target values
         :return : best feature and best threshold value
      """
      total_actual_features = x.shape[1]
      if self.max_features is not None:      
         features = np.random.choice(total_actual_features, self.max_features, replace=False)           
      else:
         features = range(total_actual_features)

      min_variance = -float('inf')
      best_feature, best_threshold = None, None

      for feature in features:
         feature_values = x[:, feature]
         thresholds = np.unique(feature_values)

         for threshold in thresholds:
            left_child_node = y[feature_values <= threshold]
            right_child_node = y[feature_values > threshold]
            reduced_variance = self.variance_reduction(y, left_child_node, right_child_node)
            if reduced_variance > min_variance:
               min_variance = reduced_variance
               best_feature = feature
               best_threshold = threshold

      return best_feature, best_threshold  

   def calculate_leaf_value(self, y):
      """Calculate the predicted value for a leaf node (mean of target values)
         :param y : target values of the current node
         :return : predicted value for the leaf node
      """
      return np.mean(y)

   def variance_reduction(self, y, left_child_node, right_child_node):
      """Calculate the reduction in variance by splitting the dataset. Variance reduction evaluates
         how much a split of a node decreases the variance of the target variable within the child
         nodes created by the split.
         Formula: VarRed = variance of parent node - weighted average variance of child nodes
         :param y : target values
         :param left_child_node : left child node of the parent node
         :param right_child_node : right child node of the parent node
         :return : reduction in variance
      """
      if len(left_child_node) == 0 or len(right_child_node) == 0:
         return 0
      
      parent_variance = np.var(y)
      left_variance = np.var(left_child_node)
      right_variance = np.var(right_child_node)
      childern_variance = (left_variance * len(left_child_node) / len(y)) + (right_variance * len(right_child_node) / len(y))
      reduced_variance = parent_variance - childern_variance
      return reduced_variance
   
   def make_predictions(self, x_test):
      """Predict the target values for the given data using the trained model
         :param x_test : test input features
         :return : array of predicated target value
      """
      return np.array( [self.traverse_tree(i, self.root_node) for i in x_test] )
      
   def traverse_tree(self, x, node):
      """Traverse through the tree to predict the target value for a single input.
         :param x : single input data point
         :param node : current node in the tree (traversing begins from the root_node node)
         :return : predicted target value
      """
      if node.value is not None:
         return node.value     
      if x[node.feature] <= node.threshold:
         return self.traverse_tree(x, node.left)  
      else:   
         return self.traverse_tree(x, node.right)
      
   def mse(self, y_test, y_predicted):
      """Determine the Mean Squared Error (MSE) between the true and predicted values.
         :param y_test : true values of the target variable
         :param y_predicted : predicted values of target variable
         :return : MSE of the model
      """
      return np.mean(np.square(y_test - y_predicted))

# EXAMPLE USAGE ON THE RED WINE QUALITY DATA #
if __name__ == "__main__":
   data = pd.read_csv("datasets\decision-tree-regression-data.csv")
   X = data.iloc[:, :-1].values
   y = data['quality'].values
   x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

   # Training the decision tree model without using sklearn library 
   regressor = Decision_Tree_Regressor(max_depth=50, min_samples_split=3, max_features=None)
   regressor.fit(x_train, y_train)

   # Training the Sklearn decision tree model
   sklearn_regressor = DecisionTreeRegressor(max_depth=70, min_samples_split=3, max_features=None)
   sklearn_regressor.fit(x_train, y_train)

   # Predictions from both the models
   our_predictions = regressor.make_predictions(x_test)
   print("\nOur Predictions:\n", our_predictions)
   sklearn_prediction = sklearn_regressor.predict(x_test)
   print("\nSklearn Predictions:\n", sklearn_prediction)

   # MSE from both the models
   our_mse = regressor.mse(y_test, our_predictions)
   print("\nMean Squared Error from Our model:", round(our_mse, 5))
   sklearn_mse = mean_squared_error(y_test, sklearn_prediction)
   print("Mean Squared Error from Slearn model:", round(sklearn_mse, 5))

   #----If the data has categorical features, we need to handle them as well by performing encoding methods before train-test-split----#
   """Identify categorical features
      Perform Encoding techniques like :
      One-hot encoding (Creates binary vales 0/1)
      Label Encoding (assigns a number to each category, but considers ordering)
      Ordinal Encoding (Similar to label encoding, but only use if categories have a natural order (like low, medium, high).)
   """