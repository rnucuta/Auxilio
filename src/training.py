# python3 linux
# Project:  Keshav
# Filename: training.py
# Author:   Raymond G. Nucuta (rnucuta@gmail.com)

"""
training: Script that trains on data from gtrends_neutralizer.
		  Outputted model can be found in /models. This model can be used by 
		  inference.py to run inference on just a particular date.
"""

# inspiration from https://www.dataquest.io/blog/sci-kit-learn-tutorial/
# choose algorithm here: https://scikit-learn.org/stable/tutorial/machine_learning_map/
# -Naive Bayes
# -Linear SVC
# -K-Neighbours Classifier

import pandas
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

le = preprocessing.LabelEncoder()

# transform each date into just the month number


# preprocess all the data into numberical representations

# split training and validation data

# import a training algorithm and train and save weights