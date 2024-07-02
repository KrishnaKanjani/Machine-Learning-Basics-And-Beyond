"""Random forest for classification is an ensemble learning algorithm that works by constructing multiple decision trees during training and outputs the mode (majority voting) of the classes. Each
   tree is built using a random subset of the training data and a random subset of the features. This randomness is achieved by a technique known as Bootstrap Sampling, which  helps to prevent 
   overfitting and makes the model robust against noise. During prediction, each tree independently classifies the input data. The final prediction is determined by aggregating the votes of all the
   trees using majority voting. As the random forest is built on the top of decision trees, we will import the decision tree classifier from our implementation and the rest will be done from scratch. 
   So, Let's begin!! 
   NOTE: The sklearn library is used just to import the Wine Dataset, prepare the training and testing datasets and to compare our results with the sklearn RandomForestClassifier.
"""

# IMPORTING NECESSARY LIBRARIES #
from decision_tree_for_classification import Decision_Tree_Classifier   
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier     

# RANDOM FOREST CLASSIFIER CLASS #
"""The Random_Forest_Classifier class is responsible for doing the following: (1) Handle the no of trres, maximum depth, minimum samples and max features for splits (pre pruning); 
   (2) Perform bootstrap sampling to generate random subsets of the training data; (3) Fit the model to training data by constructing multiple decision trees, each trained on a 
   bootstrap sample of the data; (4) Aggregate predictions from individual decision trees using majority voting to make final predictions; (5) Calculate accuracy of the model.
"""
class Random_Forest_Classifier:
   def __init__(self, no_of_trees=10, max_depth=10, min_samples_split=2, max_features=None):
      self.no_of_trees = no_of_trees
      self.max_depth = max_depth
      self.min_samples_split = min_samples_split
      self.max_features = max_features
      self.trees = []   # list to store the individual decision trees in the forest

   def fit(self, x, y):
      """Fit the random forest classifier to the training data. This method trains a forest of decision trees using bootstrapped 
         samples of the training data.
         :param x : feature matrix
         :param y : class labels 
      """
      self.trees = []
      for _ in range(self.no_of_trees):
         tree = Decision_Tree_Classifier(max_depth = self.max_depth, 
                                         min_samples_split = self.min_samples_split,
                                         max_features = self.max_features)
         x_sample, y_sample = self.bootstrap_sampling(x, y)
         tree.fit(x_sample, y_sample)
         self.trees.append(tree)

   def bootstrap_sampling(self, x, y):
      """Perform bootstrap sampling on the given data. Bootstrap sampling involves randomly sampling with replacement from the 
         original dataset. This method creates new samples where each sample is of the same size as the original dataset, but
         some instances may be duplicated (sampled more than once) and others may not be included at all. This helps to prevent
         overfitting and makes the model robust against noise.
         :param x : feature matrix
         :param y : class labels 
         :return x_sample, y_sample : bootstrapped sample of input features and corresponding labels
      """
      no_of_samples = x.shape[0]
      ids = np.random.choice(no_of_samples, no_of_samples, replace=True)
      return x[ids], y[ids]

   def make_predictions(self, x_test):
      """Predict the class labels for the given data using the trained model
         :param x_test : test input features
         :return : array of predicated class label
      """
      tree_predictions = np.array([tree.make_predictions(x_test) for tree in self.trees])  # predictions from each tree
      tree_predictions = tree_predictions.transpose()  # swap axes to have tree predictions as rows and instances as columns
      final_predictions = [np.bincount(prediction).argmax() for prediction in tree_predictions]  # for each instance, compute most frequent class label among all predictions from the trees.
      return np.array(final_predictions)
    
   def accuracy(self, y_test, y_predicted):
      """Determine how accurate our model works.
         :param y_test : true values of the class labels
         :param y_predicted : predicted values of class labels
         :return : accuracy (in %) of the model
      """
      return np.mean(y_test == y_predicted)*100
   
# EXAMPLE USAGE ON THE WINE DATASET #
if __name__ == "__main__":
   data = datasets.load_wine()
   x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)
   print("True class labels:\n", y_test)

   # Training the random forest model without using sklearn library 
   rfc = Random_Forest_Classifier(no_of_trees=25)
   rfc.fit(x_train, y_train)

   # Training the Sklearn random forest model
   sklearn_rfc = RandomForestClassifier(n_estimators=25, random_state=42)
   sklearn_rfc.fit(x_train, y_train)

   # Predictions from both the models
   our_predictions = rfc.make_predictions(x_test)
   print("\nOur Predictions:\n", our_predictions)
   sklearn_prediction = sklearn_rfc.predict(x_test)
   print("\nSklearn Predictions:\n", sklearn_prediction)

   # Accuracy from both the models
   our_accuracy = rfc.accuracy(y_test, our_predictions) 
   print(f"\nAccuracy of our model: {round(our_accuracy, 3)}%")
   sklearn_accuracy = accuracy_score(y_test, sklearn_prediction)*100
   print(f"Accuracy of Sklearn model: {round(sklearn_accuracy, 3)}%")