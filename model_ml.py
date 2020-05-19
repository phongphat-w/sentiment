# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
@create: 2020-04-07
"""
#=====================
#Download and install package

#Require installation of these package by:
#1) command line or 
#2) the parent object (caller object: *.py, *.ipynb)

#---> pip install xgboost
#---> pip install --upgrade xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from scipy.stats import uniform


class model_ml():

	def __init__(self, X=None, y=None, base_estimator=None):
		
		self.model_return = {}
		
		self.X_train = X
		self.y_train = y  
		
		self.base_estimator = base_estimator
		
		
	def logistic_base(self):		
		print ( "\n logistic_baseline() is activated...\n" )

		model_logr = {}
		
		model_logr["tuned_model"] = LogisticRegression(random_state=0).fit(self.X_train, self.y_train)		
		
		return model_logr


	def logistic_regression(self):		
		print ( "\n logistic_regression() is activated...\n" )
	
		model_logr = {}
		
		model_logr["hparams"] = {"C": uniform(loc=0, scale=4)}
		model_logr["model"] = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced")
		model_logr["rcv"] = RandomizedSearchCV(model_logr["model"], model_logr["hparams"], random_state=0)
		model_logr["rsearch"] = model_logr["rcv"].fit(self.X_train, self.y_train)
		
		"""Finding the best hyperparameters """		
		model_logr["rtuned_hparams"] = model_logr["rsearch"].best_params_
		
		
		"""Grid search """		
		C = model_logr["rtuned_hparams"]["C"]
		model_logr["gparams"] = {"C": [C - (C*i/10) for i in range(5)] + [C + (C*i/10) for i in range(1, 5)]}
		model_logr["gcv"] = GridSearchCV(model_logr["model"], model_logr["gparams"], cv=10)
		model_logr["gcv"].fit(self.X_train, self.y_train)
		model_logr["tuned_model"] = model_logr["gcv"].best_estimator_
		
		return model_logr


	def gaussian_nb(self):
		print ( "\n gaussian_nb() is activated...\n" )
		
		model_gnb = {}
		
		model_gnb["model"] = GaussianNB()
		model_gnb["tuned_model"] = model_gnb["model"].fit(self.X_train, self.y_train)
				
		return model_gnb


	def random_forest(self):
		print ( "\n random_forest() is activated...\n" )

		model_rtn = {}

		model_rtn["hparams"] = {"max_depth": [2, 4, 6], "min_samples_split": [5, 10]}
		model_rtn["model"] = RandomForestClassifier(random_state=0, class_weight="balanced")
		model_rtn["rcv"] = RandomizedSearchCV(model_rtn["model"], model_rtn["hparams"], random_state=0)
		model_rtn["rsearch"] = model_rtn["rcv"].fit(self.X_train, self.y_train)

		"""Finding the best hyperparameters """
		model_rtn["rtuned_hparams"] = model_rtn["rsearch"].best_params_

		"""Grid search """
		max_depth = model_rtn["rtuned_hparams"]["max_depth"]
		min_samples_split = model_rtn["rtuned_hparams"]["min_samples_split"]
		model_rtn["gparams"] = {"max_depth": [max_depth - 1, max_depth, max_depth + 1], 
								"min_samples_split": [min_samples_split - 1, min_samples_split, min_samples_split + 1]
								}
		model_rtn["gcv"] = GridSearchCV(model_rtn["model"], model_rtn["gparams"], cv=10)
		model_rtn["gcv"].fit(self.X_train, self.y_train)
		model_rtn["tuned_model"] = model_rtn["gcv"].best_estimator_

		return model_rtn


	def svm(self):
		print ( "\n svm() is activated...\n" )

		model_svm = {}

		model_svm["hparams"] = {'C': [10, 100]}
		model_svm["model"] = SVC(kernel="linear", probability = True)		
		model_svm["rcv"] = RandomizedSearchCV(model_svm["model"], model_svm["hparams"], random_state=0) 
		
		print ( "\n RandomizedSearchCV fitting...\n" )
		model_svm["rsearch"] = model_svm["rcv"].fit(self.X_train, self.y_train)
		print ( "\n RandomizedSearchCV finish fitting.\n" )
		
		
		"""Finding the best hyperparameters """	
		model_svm["rtuned_hparams"] = model_svm["rsearch"].best_params_
		
		
		"""Grid search """		
		C = model_svm["rtuned_hparams"]["C"]
		model_svm["gparams"] = {"C": [C - (C*i/10) for i in range(5)] + [C + (C*i/10) for i in range(1, 5)]}
		model_svm["gcv"] = GridSearchCV(model_svm["model"], model_svm["gparams"], cv=5, iid = False, n_jobs = -1, verbose = True)
		model_svm["gcv"].fit(self.X_train, self.y_train)
		model_svm["tuned_model"] = model_svm["gcv"].best_estimator_
		
		return model_svm


	def xgboost(self):
		print ( "\n xgboost() is activated...\n" )
		
		model_xgb = {}

		model_xgb["model"] = XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
								max_depth = 5, alpha = 10, n_estimators = 10)								
		model_xgb["tuned_model"] = model_xgb["model"].fit(self.X_train, self.y_train)
		
		return model_xgb