# python3 linux
# Project:  Trendsy
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


# BEST MODEL: KNEIGHBORS CLASSIFIER
# try svc/ensemble classifiers next


import argparse
from datetime import datetime
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import joblib
import copy
import os
import numpy as np

# le = preprocessing.LabelEncoder()

def categorizer(number):
	if number<30:
		return 0
	if number>70:
		return 2
	else: 
		return 1

def incidence_categorizer(number):
	if number>94:
		return 1
	else:
		return 0

# transform each date into just the month number
def load_data(csv_name):
	df = pd.read_csv('../dumps/'+csv_name)
	cols = [col for col in df.columns if col not in ['Date','DiseaseIncidence', 'AdjustedDiseaseIncidence', 'LOW', 'MEDIUM']]
	# colss = [col for col in df.columns if col not in ['Date', 'LOW', 'MEDIUM']]
	for i in range(len(df.index)):
		temp_date=datetime.strptime(df.at[i,"Date"], '%m/%d/%Y %M:%S')
		df.at[i,'Date']=int(temp_date.month)
		df.at[i, 'DiseaseIncidence']=categorizer(df.at[i, 'DiseaseIncidence'])
		for col in cols:
			df.at[i, col]=categorizer(int(df.at[i, col]))
	# for i in range(len(df.index)):
	# 	for col in colss:
	# 		df.at[i, col]-=1
	return df

# preprocess all the data into target and features
def preprocess(month_df):
	# select columns other than 'DiseaseIncidence'
	cols = [col for col in month_df.columns if col not in ['DiseaseIncidence', 'AdjustedDiseaseIncidence', 'LOW', 'MEDIUM']]
	# dropping the 'DiseaseIncidence' column
	data = month_df[cols]
	print(data)
	#assigning the DiseaseIncidence column as target
	target = month_df['DiseaseIncidence']
	return data,target

# split training and validation data
def split_data(data, target):
	return train_test_split(data,target, test_size = 0.2, random_state=42)

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
	best_model=None
	best_variance=0
	best_r2=0
	for i in range(10):
		reg = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
		model=reg.fit(data_train, target_train)
		pred=model.predict(data_test)
		if(explained_variance_score(target_test, pred)>best_variance):
			best_model=copy.deepcopy(model)
			best_variance=float(explained_variance_score(target_test, pred))
			best_r2=float(r2_score(target_test, pred))
	print("ElasticNet accuracy: ",best_variance)
	print("ElasticNet R^2: ",best_r2)
	return best_model, best_variance

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

def svc_train(data_train, data_test, target_train, target_test):
	from sklearn.svm import LinearSVC
	from sklearn.metrics import accuracy_score
	best_model=None
	best_variance=0
	for i in range(10):
		reg = LinearSVC(random_state=0)
		model=reg.fit(data_train, target_train)
		pred=model.predict(data_test)
		if(accuracy_score(target_test, pred, normalize = True)>best_variance):
			best_model=copy.deepcopy(model)
			best_variance=float(accuracy_score(target_test, pred, normalize = True))
	print("LinearSVC accuracy: ",best_variance)
	return best_model, best_variance

# def KNeighbors_train(data_train, data_test, target_train, target_test):
# 	from sklearn.neighbors import KNeighborsClassifier
# 	from sklearn.neighbors import NeighborhoodComponentsAnalysis
# 	from sklearn.metrics import accuracy_score
# 	nca = NeighborhoodComponentsAnalysis(random_state=42)
# 	nca.fit(data_train, target_train)
# 	best_model=None
# 	best_variance=0
# 	for i in range(10):
# 		reg = KNeighborsClassifier(n_neighbors=17)
# 		model=reg.fit(data_train, target_train)
# 		# pred=reg.predict(data_test)
# 		if(reg.score(nca.transform(data_test), target_test)>best_variance):
# 			best_model=copy.deepcopy(model)
# 			best_variance=float(reg.score(nca.transform(data_test), target_test))
# 	print ("KNeighbors accuracy score : ", best_variance)
# 	return best_model, best_variance

def KNeighbors_train(data_train, data_test, target_train, target_test):
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import accuracy_score
	best_model=None
	best_variance=0
	for i in range(10):
		reg = KNeighborsClassifier(n_neighbors=17)
		model=reg.fit(data_train, target_train)
		pred=reg.predict(data_test)
		if(accuracy_score(target_test, pred)>best_variance):
			best_model=copy.deepcopy(model)
			best_variance=float(accuracy_score(target_test, pred))
	print ("KNeighbors accuracy score : ", best_variance)
	return best_model, best_variance

# def KNeighbors_train(data_train, data_test, target_train, target_test):
# 	from sklearn.neighbors import KNeighborsClassifier
# 	from sklearn.metrics import accuracy_score
# 	from sklearn.model_selection import GridSearchCV
# 	knn2 = KNeighborsClassifier()
# 	param_grid = {'n_neighbors': np.arange(1, 25)}
# 	knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
# 	knn_gscv.fit(data_train, target_train)
# 	print(knn_gscv.best_params_)
# 	print(knn_gscv.best_score_)
# 	return knn2, 0.1

def choose_model(t_type, data_train, data_test, target_train, target_test):
	if t_type=="lasso":
		return lasso_train(data_train, data_test, target_train, target_test)
	elif t_type=="elastic_net":
		return elasticNet_train(data_train, data_test, target_train, target_test)
	elif t_type=="ridge":
		return ridge_train(data_train, data_test, target_train, target_test)
	elif t_type=="svm_linear":
		return svm_train(data_train, data_test, target_train, target_test)
	elif t_type=="svm_rbf":
		return svm_train(data_train, data_test, target_train, target_test, 'rbf')
	elif t_type=="svc":
		return svc_train(data_train, data_test, target_train, target_test)
	elif t_type=="KNeighbors":
		return KNeighbors_train(data_train, data_test, target_train, target_test)
	else:
		print("Incorrect argument for --training_type. Try again.")


if __name__ == '__main__':
    # Parse command line arguments.

    #default command: python3 training.py --disease_freq "valley fever_weeklyData.csv" --training_type "elastic_net"

    parser = argparse.ArgumentParser(description=__doc__)
    # Data files/directories.
    parser.add_argument('--disease_freq', required=True, \
                        help='name of .csv file with weekly trends/incidence data that is in /dumps')
    parser.add_argument('--training_type', required=True, \
                        help=r'options: {"lasso", elastic_net, ridge, svm_linear, svm_rbf, svc, KNeighbors}')
    args = parser.parse_args()

    loaded_df=load_data(args.disease_freq)

    features_data,target_data=preprocess(loaded_df)

    data_train, data_test, target_train, target_test=split_data(features_data,target_data)

    trained_model, trained_variance = choose_model(args.training_type, data_train, data_test, target_train, target_test)

    if trained_model!=None:
    	# log_time=datetime.now() str(log_time)[:str(log_time).index('.')].replace(':', '.').replace(" ", "--")+"--"
    	logs_file_location=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'models',args.training_type+" acc "+str(round(trained_variance,3))+".sav"))
    	joblib.dump(trained_model, logs_file_location)
