"""Real world datasets often contain features that are varying in degrees of magnitude, range and units. Therefore, in order for machine learning models to interpret these 
   features on the same scale, feature scaling is needed. Feature scaling is a preprocessing technique that transforms feature values to a similar scale, ensuring all features
   contribute equally to the model. This helps improve the performance and training stability of models. It is crucial for algorithms that rely on distance measurements or gradient
   descent optimization. There are several ways to do feature scaling. Let's explore few of them! We will try not to use any inbuilt libraries to demonstrate its functionality. 
   NOTE: The sklearn library is used just to compare our results with the built-in sklearn scaling functions.
   Let's begin !! 
"""

import numpy as np

# 1. Absolute Maximum Scaling
def abs_max_scaling(x):
    """Absolute Maximum Scaling is the easiest scaling technique that scales the data to its maximum value i.e it divides every 
       data point by the maximum value of the feature(x). This technique is useful when we want to normalize by the maximum 
       absolute value, regardless of distribution. The result of this scaling lies within the range of -1 to 1.
       Formula: x_scaled = x / x_max
    """
    x_max = np.max(x)
    x_scaled = x / x_max
    return x_scaled

# 2. Min-Max Scaling (Normalization)
def min_max_scaling(x):
    """Min-Max Scaling is a scaling technique in which values are shifted and rescaled so that they end up ranging between 0 
       and 1. This technique is sensitive to outliers. This scaler is a good choice for us to use if the standard deviation is 
       small and when a distribution is normal.
       Formula: x_scaled = (x - x_min) / (x_max - x_min)
       Here, x_max and x_min are the maximum and the minimum values of the feature, respectively
    """
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled

# 3. Mean Normalization 
def mean_normalization(x):
    """Mean normalization is a scaling technique in which values are shifted and scaled such that the mean is 0 and values range
       between -1 and 1. This technique is used when the data has outliers and we want a mean of 0.
       Formula: x_scaled - (x - x_mean) / (x_max - x_min)
    """
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    x_mean = np.mean(x, axis=0)
    x_scaled = (x - x_mean) / (x_max - x_min)
    return x_scaled

# 4. Z-score Normalization (Standardization)
def z_score(x):
    """Z-score Normalization or Standardization is a scaling technique which scales the data to have a mean of 0 and a standard
       deviation of 1. It is useful when the distribution of the data is Gaussian or unknown. This technique is less sensitive to
       outliers. For the output range, depending on the distribution of the data, it adjusts the data based on its mean and 
       standard deviation.
       Formula: x_scaled = (x - x_mean) / sd
       where, sd (standard deviation) = sqrt(variance);  variance = 1/n * sum(square(x - x_mean))
    """
    x_mean = np.mean(x, axis=0)
    # x_sd = np.sqrt( (1/len(x)) * np.sum(np.square(x - x_mean)) )   
    x_sd = np.std(x, axis=0)      
    x_scaled = (x - x_mean) / x_sd
    return x_scaled

# 5. Robust Scaling / Robust Normalization
def robust_scaling(x):
    """Robust Scaling is a scaling method, which scales the data to have a median of 0 and adjusts for the spread of the data using
       the Interquatile Range (IQR). It is useful when dealing with data containing outliers or when the data does not follow a 
       normal distribution, ensuring that the scaled data maintains robustness against extreme values while still providing a 
       standardized form. Unlike the mean, which is sensitive to outliers, the median is robust to outliers because it is less 
       affected by extreme values.
       Formula: x_scaled = (x - x_median) / IQR
       where, IQR is a measure of spread of the data and is calculated as IQR = Q3 (75th percentile) - Q1 (25th percentile)
    """
    x_median = np.median(x, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    x_scaled = (x - x_median) / (q3 - q1)
    return x_scaled

# EXAMPLE USAGE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

data = np.array([[1, 2, 3, 4, 5], [5, 6, 7, 6, 5], [10, 11, 12, 13, 14]])

print("Absolute Maximum Scaling Results:")
print("", abs_max_scaling(data))

print("\nMean Normalization Scaling Results:")
print("", mean_normalization(data))

m_scaler = MinMaxScaler()
s_scaler = StandardScaler()  # Z-score
r_scaler = RobustScaler()

m_scaled_data = m_scaler.fit_transform(data)
s_scaled_data = s_scaler.fit_transform(data)
r_scaled_data = r_scaler.fit_transform(data)

print("\nMin Max Scaling Results:")
print(" Without sklearn library:\n", min_max_scaling(data))
print(" By using sklearn library:\n", m_scaled_data)

print("\nZ-score Scaling Results:")
print(" Without sklearn library:\n", z_score(data))
print(" By using sklearn library:\n", s_scaled_data)

print("\nRobust Scaling Results:")
print(" Without sklearn library:\n", robust_scaling(data))
print(" By using sklearn library:\n", r_scaled_data)