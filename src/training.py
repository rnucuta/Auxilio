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
# REEEEEEGRESSIONS
# Over 50 samples: yes
# Predicting a category: no
# Predicting a quantity: yes
# Less than 100K samples: yes
# Few Features should be impt: idk?
# -No: Lasso, Elastic Nets
# -Yes: Ridge Regression, SVR(kernel='linear'), SVR(kernel='rbf'), Ensemble Regressors


import argparse
import os
from datetime import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib

# le = preprocessing.LabelEncoder()

# transform each date into just the month number
def load_data(csv_name):
	df = pd.read_csv('../dumps/'+csv_name)
	for i in range(len(df.index)):
		temp_date=datetime.strptime(df.at[i,"Date"], '%m/%d/%Y %M:%S')
		df.at[i,'Date']=int(temp_date.month)
	return df

# preprocess all the data into target and features
def preprocess(month_df):
	# select columns other than 'DiseaseIncidence'
	cols = [col for col in month_df.columns if col not in ['DiseaseIncidence']]
	# dropping the 'DiseaseIncidence' column
	data = month_df[cols]
	#assigning the DiseaseIncidence column as target
	target = month_df['DiseaseIncidence']
	return data,target

# split training and validation data
def split_data(data, target):
	return train_test_split(data,target, test_size = 0.1)

# import a training algorithm and train and save weights
def lasso_train(data_train, data_test, target_train, target_test):
	from sklearn import linear_model
	from sklearn.metrics import explained_variance_score, r2_score
	reg = linear_model.Lasso(alpha=0.1)
	pred=reg.fit(data_train, target_train).predict(data_test)
	print("Lasso accuracy: ",explained_variance_score(target_test, pred))
	print("Lasso R^2: ",r2_score(target_test, pred))

def elasticNet_train(data_train, data_test, target_train, target_test):
	from sklearn import linear_model
	from sklearn.metrics import explained_variance_score, r2_score
	reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
	pred=reg.fit(data_train, target_train).predict(data_test)
	print("ElasticNet accuracy: ",explained_variance_score(target_test, pred))
	print("ElasticNet R^2: ",r2_score(target_test, pred))

def ridge_train(data_train, data_test, target_train, target_test):
	from sklearn import linear_model
	from sklearn.metrics import explained_variance_score, r2_score
	reg = linear_model.Ridge(alpha=0.5)
	pred=reg.fit(data_train, target_train).predict(data_test)
	print("Ridge Regression accuracy: ",explained_variance_score(target_test, pred))
	print("Ridge Regression R^2: ",r2_score(target_test, pred))

def svm_train(data_train, data_test, target_train, target_test, kernel_type='linear'):
	from sklearn import svm
	from sklearn.metrics import explained_variance_score, r2_score
	reg = svm.SVR(kernel=kernel_type)
	pred=reg.fit(data_train, target_train).predict(data_test)
	print("Ridge Regression accuracy: ",explained_variance_score(target_test, pred))
	print("SVM R^2: ",r2_score(target_test, pred))

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    # Data files/directories.
    parser.add_argument('--disease_freq', required=True, \
                        help='name of .csv file with weekly trends/incidence data that is in /dumps')
    # parser.add_argument('--model_file', required=True, \
    #                     help='name of model file that will after training is completed')
    args = parser.parse_args()

    loaded_df=load_data(args.disease_freq)

    features_data,target_data=preprocess(loaded_df)

    data_train, data_test, target_train, target_test=split_data(features_data,target_data)

    elasticNet_train(data_train, data_test, target_train, target_test)