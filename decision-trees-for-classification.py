"""Decision trees for classification are a machine learning model that recursively splits the data into subsets based on the value of input features, creating a tree-like structure. At each node
   of the tree, the feature that best separates the classes is chosen. For choosing the best feature for spiltting, concepts like Entropy, Gini impurity and information gain are used. This process
   of splitting continues until a stopping criteria is met, such as all samples in a node belonging to the same class or reaching a maximum tree depth. Each leaf node represents a class label.
   The model is intuitive and interpretable, but can overfit, which is solved by Pruning. In this implementation, I'll use the pruning method called Pre-Pruning and to demonstrate all these
   concepts, I have implemented the algorithm from scratch. So, Let's begin !!
   NOTE: I have to provided the description for each functionality, but to avoid the longness of code, I have kept it shorter and in brief.
   NOTE: The sklearn library is used just to import the Iris Dataset, prepare the training and testing datasets and to compare our results with the sklearn DecisionTreeClassifier.
"""

# IMPORTING NECESSARY LIBRARIES #
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# NODE CLASS #
"""The Node class will represent each node in the decision tree. It will store the information about: (1) the 'feature' to split the tree on; (2) the 'threshold' value for splitting; 
   (3) the 'left' and 'right' child nodes of each node; (4) 'value' which is a predicted class if the node is a leaf node. The default value for all will be None.
"""
class Node:
   def __init__(self, feature=None, threshold=None, left=None, right=None, *,value=None ):
      self.feature = feature
      self.threshold = threshold
      self.left = left
      self.right = right
      self.value = value

# DECISION TREE CLASSIFIER CLASS #
"""The DecisionTreeClassifier class is responsible for doing the following: (1) Handle the maximum depth, minimum samples and max features for splits (pre pruning); (2) Compute Gini 
   impurity or Entropy for a list of classes to evaluate splits; (3) Identify the best feature and threshold for splitting the data (Best Split using information gain); (4) Recursively 
   grow the tree by finding the best split; (5) Predict classes for input data by traversing the tree; (6) Calculate accuracy of the model.
"""
class Decision_Tree_Classifier: 
   def __init__(self, max_depth=None, min_samples_split=2, max_features=None, criterion='entropy'):
      self.max_depth = max_depth
      self.min_samples_split = min_samples_split
      self.max_features = max_features
      self.criterion = criterion
      self.root_node = None   # Root Node

   def fit(self, x, y):
      """Fit the decision tree classifier to the training data.
         :param x : feature matrix
         :param y : class labels
      """
      if not self.max_features:   # To make sure that the max_features does not exceed the actual no. of features in our dataset.
         self.max_features = x.shape[1] 
      else:
         self.max_features = min(x.shape[1], self.max_features)

      self.root_node = self.grow_tree(x, y)

   def entropy(self, y):
      """Calculate the entropy of the node/data. Entropy is the measure used to measure the uncertainty in the data i.e
         determine the best way to split the data by quantifying the impurity of a node. Low entropy indicates that 
         most data points belong to one class. 
         Formula: entropy(data) = -sum( pi*log2(pi) ) 
         where, pi = probability of class i in the data = No. of occurrences / total no. occurrences
         :param y : Class Labels
         :return : Entropy of the node
      """
      if len(y) == 0:
        return 0
      epsilon = 1e-9
      no_of_occurrences = np.bincount(y)
      probabilities = no_of_occurrences / len(y)
      entropy = -np.sum(probabilities * np.log2(probabilities + epsilon))
      return entropy

   def gini(self, y):
      """Calculate the Gini impurity of the node/data. Gini impurity is another measure of node impurity. It gives the 
         probability of missclassifying the new instance. Low gini indicates that most data points belong to one class. 
         Formula: gini(data) = 1 - sum( pi^2 )
         where, pi = probability of class i in the data = No. of occurrences / total no. occurrences
         :param y : Class Labels
         :return : Gini index of the node
      """
      if len(y) == 0:
         return 0
      no_of_occurrences = np.bincount(y)
      probabilities = no_of_occurrences / len(y)
      gini_impurity = 1 - np.sum(np.square(probabilities))
      return gini_impurity

   def information_gain(self, y, left_child_node, right_child_node):
      """Calculate the Information gain for a given feature and threshold. Information gain is a measure used to find 
         the feature to be used for splitting. It is calculated with the help of entropy/gini impurity. Higher information 
         gain of a feature indicates that the feature is more informative and should be used for splitting.
         Formula: information_gain = impurity of parent node - weighted avg. impurity of child nodes"
         :param y : class labels
         :param left_child_node : left child node of the parent node
         :param right_child_node : right child node of the parent node
         :return : The information gain obtained by splitting the node based on the given feature and threshold.
      """
      if len(left_child_node) == 0 or len(right_child_node) == 0:
         return 0
      
      parent_impurity = self.entropy(y) if self.criterion == 'entropy' else self.gini(y)   # impurity of parent node      
      left_impurity= self.entropy( y[left_child_node] ) if self.criterion == 'entropy' else self.gini( y[left_child_node] )  # impurity of left child node
      right_impurity =  self.entropy( y[right_child_node] ) if self.criterion == 'entropy' else self.gini( y[right_child_node] )  # impurity of right child node    
      children_impurity = ( left_impurity * len(left_child_node) / len(y) ) + ( right_impurity * len(right_child_node) / len(y) )  # total weighted avg. impurity of child nodes  
      information_gain = parent_impurity - children_impurity     
      return information_gain

   def best_split(self, x, y):
      """Find the best feature and threshold for splitting the data, using information gain.
         :param x : feature matrix
         :param y : class labels
         :return : best feature and best threshold value
      """
      total_actual_features = x.shape[1]
      if self.max_features is not None:      # randomly select the features from the available features for splitting
         features = np.random.choice(total_actual_features, self.max_features, replace=False )           
      else:
         features = range(total_actual_features)

      best_info_gain = -1
      best_feature, best_threshold = None, None

      for feature in features:
         feature_values = x[:, feature]
         thresholds = np.unique(feature_values)    # unique values of the current feature

         for threshold in thresholds:
            left_child_node = np.where(feature_values <= threshold)[0]   # split the node with current feature & threshold, to create left child node 
            right_child_node = np.where(feature_values > threshold)[0]   # split the node with current feature & threshold, to create right child node 
            info_gain = self.information_gain(y, left_child_node, right_child_node)   # calculate the information gain for splitting current feature at the 'thresthold' 
            # check if the info_gain is greater than the best_info_gain
            if info_gain > best_info_gain:  
               best_info_gain = info_gain
               best_feature = feature
               best_threshold = threshold

      return best_feature, best_threshold
   
   def grow_tree(self, x, y, depth=0):
      """Recursively build the decision tree
         :param x : feature matrix
         :param y : class labels
         :param depth : current depth of the tree
         :returns (Node) : root node of the subtree
      """
      no_samples_per_class = np.bincount(y)   # no of samples for each class in the current node
      predicted_class = np.argmax(no_samples_per_class)   # class with the highest count (mode)
      node = Node(value = predicted_class)   # create a node with predicted class value

      # Check the stopping criteria for recursion
      # 1. Return leaf node if the node has exceeded the parameter values
      if depth >= self.max_depth or x.shape[0] < self.min_samples_split:
         return node         
      # 2. Else, find the best split i.e identify the best feature and threshold for splitting the data
      else:            
         best_feature, best_threshold = self.best_split(x, y)
         # Return leaf node if no best feature and threshold is found
         if best_feature is None or best_threshold is None:
            return node
         # Else, split the node with best feature and threshold found, to create child nodes 
         else:      
            left_child_node = np.where(x[:, best_feature] <= best_threshold)[0]
            right_child_node = np.where(x[:, best_feature] > best_threshold)[0]   

            if len(left_child_node) == 0 or len(right_child_node) == 0:
               return node               
            
            left = self.grow_tree( x[left_child_node, :], y[left_child_node], depth + 1 )
            right = self.grow_tree( x[right_child_node, :], y[right_child_node], depth + 1 )
            return Node(best_feature, best_threshold, left, right)
      
   def traverse_tree(self, x, node):
      """Traverse through the tree to predict the class for a single input
         :param x : single input data point
         :param node : current node in the tree (traversing begins from the root_node node)
         :return : predicted class label
      """
      # Check Purity i.e. check if the current node is a leaf node i.e our prediction 
      if node.value is not None:     
         return node.value 
      
      # Decide on which sight of the tree is to be traversed next for prediction (left or right)
      if x[node.feature] <= node.threshold:
         return self.traverse_tree(x, node.left)
      else:
         return self.traverse_tree(x, node.right)
 
   def make_predictions(self, x_test):
      """Predict the class labels for the given data using the trained model
         :param x_test : test input features
         :return : array of predicated class label"""
      return np.array( [self.traverse_tree(i, self.root_node) for i in x_test] )
   
   def accuracy(self, y_test, y_predicted):
      """Determine how accurate our model works.
         :param y_test : true values of the class labels
         :param y_predicted : predicted values of class labels
         :return : accuracy (in %) of the model"""
      return np.mean(y_test == y_predicted)*100


# EXAMPLE USAGE ON THE IRIS DATA #
data = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
print("True class labels:\n", y_test)

# Training the decision tree model without using sklearn library 
tree = Decision_Tree_Classifier(max_depth=50, min_samples_split=3, max_features=4, criterion='entropy')
tree.fit(x_train, y_train)

# Training the Sklearn decision tree model
sklearn_tree = DecisionTreeClassifier(max_depth=50, min_samples_split=3, max_features=4, criterion='entropy') 
sklearn_tree.fit(x_train, y_train)

# Predictions and Accuracy from our model
our_predictions = tree.make_predictions(x_test)
print("\nOur Predictions:\n", our_predictions)
our_accuracy =  tree.accuracy(y_test, our_predictions) 
print(f"Accuracy of our model: {round(our_accuracy, 3)}%")

# Predictions and Accuracy from Sklearn model
sklearn_prediction = sklearn_tree.predict(x_test)
print("\nSklearn Predictions:\n", sklearn_prediction)
sklearn_accuracy = accuracy_score(y_test, sklearn_prediction)*100
print(f"Accuracy of Sklearn model: {round(sklearn_accuracy, 3)}%")