# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
@create: 2020-04-07
"""

import pandas as pd
import matplotlib.pyplot as plt
import model_ml as mml
import scipy.sparse as sparse

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, roc_curve, roc_auc_score

#reduce feature
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class model_sentiment():

	def __init__(self, dict_param = None):
		#
		self.dict_param = dict_param
		self.k_fold = 10
		self.target_num_feature = 10
		#
		self.X_array = self.dict_param["data_tfidf_vectorizer"].toarray()
		self.y_array = self.dict_param["polarity"].to_numpy()
		#
		self.X_array_pca = self.dict_param["data_tfidf_vectorizer"].toarray()		
		#
		self.dict_kf_model_lgr = {}
		self.dict_kf_model_gnb = {}
		self.dict_kf_model_rf = {}
		self.dict_kf_model_svm = {}		
		self.dict_kf_model_xgb = {}	
		#
		#================
		#Start ML process
		#
		self.reduce_feature()
		self.kfold_train_test()
		self.compare_final_result()
		
		
	def reduce_feature(self):
		#
		print ( "\n reduce_feature() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("\nBefore reduce feature")
		#plt.spy(self.X_array, markersize=1)
		#
		print("\n\nself.X_array_pca")
		print(self.X_array_pca)
		#	
		#		
		#Use standard scaler to normalize the features
		scaler = StandardScaler()
		self.X_array_pca = scaler.fit_transform(self.X_array_pca)
		#
		pca = PCA(n_components=self.target_num_feature)
		self.X_array_pca = pca.fit_transform(self.X_array_pca)
		#
		print("\nexplained_variance_ratio_:")
		print(pca.explained_variance_ratio_)
		#
		print("\nsingular_values_:")
		print(pca.singular_values_)
		#
		print("\nAfte reduce feature")
		print("\n\nself.X_array_pca, shape: ", self.X_array_pca.shape)
		print(self.X_array_pca)


	def kfold_train_test(self):
		#
		print ( "\n kfold_train_test() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#		
		print("\X_array_pca:\n")
		print(self.X_array_pca)
		
		print("\ny_array:")
		print(self.y_array)
		#
		X = self.X_array_pca		
		y = self.y_array
		#
		#Configure 10-folds cross validation
		kf = KFold(n_splits=self.k_fold, shuffle=False, random_state=None)
		#
		i = 0
		dict_kf_model_param = {}		
		#
		for train_index, test_index in kf.split(X):		
			#
			print("\nKFold: ", i)
			#
			#print("\nTRAIN:\n", train_index, "\n\nTEST:\n", test_index)
			#
			X_train, X_test = X[train_index], X[test_index]			
			y_train, y_test = y[train_index], y[test_index]
			#
			print("\nX_train:\n", X_train, "\nShape: ", X_train.shape)
			print("\nX_test:\n", X_test, "\nShape: ", X_test.shape)
			#
			print("\ny_train:\n", y_train, "\nShape: ", y_train.shape)
			print("\ny_test:\n", y_test, "\nShape: ", y_test.shape)
			#
			dict_kf_model_param["kf_id"] = i
			dict_kf_model_param["X_train"] = X_train
			dict_kf_model_param["X_test"] = X_test
			dict_kf_model_param["y_train"] = y_train
			dict_kf_model_param["y_test"] = y_test
			
			self.train_model(dict_kf_model_param)
			#
			i += 1



	def train_model(self, dict_kf_model_param):
		#
		print( "\n train_model() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		self.logistic_baseline(dict_kf_model_param)
		self.gaussian_nb(dict_kf_model_param)
		self.random_forest(dict_kf_model_param)
		self.svm(dict_kf_model_param)
		self.xgboost(dict_kf_model_param)
	
	def logistic_baseline(self, dict_kf_model_param):
		#
		print( "\n logistic_baseline() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("nKFold round: ", dict_kf_model_param["kf_id"])
		#
		model_logr_baseline = {}
		model_logr_baseline = mml.model_ml(X = dict_kf_model_param["X_train"], y = dict_kf_model_param["y_train"]).logistic_regression()

		"""Intercept and Coefficients"""

		print( "\n Intercept \n" + str(model_logr_baseline["tuned_model"].intercept_[0]) )
		print( "\n Coefficients \n" + str(model_logr_baseline["tuned_model"].coef_[0]) )

		"""Evaluate and predict"""

		model_logr_baseline["pop_polarity"] = model_logr_baseline["tuned_model"].predict_proba(dict_kf_model_param["X_test"])[:, 1]
		model_logr_baseline["yhat"] = model_logr_baseline["tuned_model"].predict(dict_kf_model_param["X_test"])

		model_logr_baseline["con_matrix"] = confusion_matrix(dict_kf_model_param["y_test"], model_logr_baseline["yhat"], labels=[0, 1])
		model_logr_baseline["class_report"] = classification_report(dict_kf_model_param["y_test"], model_logr_baseline["yhat"])

		model_logr_baseline["roc"] = roc_curve(dict_kf_model_param["y_test"], model_logr_baseline["pop_polarity"], pos_label=1)
		model_logr_baseline["auc"] = round(roc_auc_score(dict_kf_model_param["y_test"], model_logr_baseline["pop_polarity"]), 2)

		print( "\n pop_polarity" )
		print( model_logr_baseline["pop_polarity"] )

		print( "\n yhat" )
		print( model_logr_baseline["yhat"] )

		print( "\n auc" )
		print( model_logr_baseline["auc"] )


		"""Confusion Matrix"""

		print( """\n model_logr_baseline["con_matrix"]""" )
		print( model_logr_baseline["con_matrix"] )

		# True Positives
		model_logr_baseline["TP"] = model_logr_baseline["con_matrix"][1, 1]

		# True Negatives
		model_logr_baseline["TN"] = model_logr_baseline["con_matrix"][0, 0]

		# False Positives
		model_logr_baseline["FP"] = model_logr_baseline["con_matrix"][0, 1]

		# False Negatives
		model_logr_baseline["FN"] = model_logr_baseline["con_matrix"][1, 0]

		model_logr_baseline["accuracy"] = (model_logr_baseline["TP"] + model_logr_baseline["TN"]) / float(model_logr_baseline["TP"] + model_logr_baseline["TN"] + model_logr_baseline["FP"] + model_logr_baseline["FN"])
		model_logr_baseline["recall"] = model_logr_baseline["TP"] / float(model_logr_baseline["TP"] + model_logr_baseline["FN"])
		model_logr_baseline["specificity"] = model_logr_baseline["TN"] / float(model_logr_baseline["TN"] + model_logr_baseline["FP"])
		model_logr_baseline["precision"] = model_logr_baseline["TP"] / float(model_logr_baseline["TP"] + model_logr_baseline["FP"])

		print("\n Model performance calculate from confusion matrix:\n" )
		print("accuracy: ", model_logr_baseline["accuracy"] )
		print("recall: ",model_logr_baseline["recall"] )
		print("specificity: ",model_logr_baseline["specificity"] )
		print("precision: ",model_logr_baseline["precision"] )

		print ( """\n model_logr_baseline["class_report"]""" )
		print ( model_logr_baseline["class_report"] )
		
		self.dict_kf_model_lgr[dict_kf_model_param["kf_id"]] = model_logr_baseline
	
	
	def gaussian_nb(self, dict_kf_model_param):
		#
		print( "\n gaussian_nb() - Gaussian Naive Bayes is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("nKFold round: ", dict_kf_model_param["kf_id"])
		#
		model_gnb = {}
		model_gnb = mml.model_ml(X = dict_kf_model_param["X_train"], y = dict_kf_model_param["y_train"]).gaussian_nb()

		"""Evaluate and predict"""

		model_gnb["pop_polarity"] = model_gnb["tuned_model"].predict_proba(dict_kf_model_param["X_test"])[:, 1]
		model_gnb["yhat"] = model_gnb["tuned_model"].predict(dict_kf_model_param["X_test"])

		model_gnb["con_matrix"] = confusion_matrix(dict_kf_model_param["y_test"], model_gnb["yhat"], labels=[0, 1])
		model_gnb["class_report"] = classification_report(dict_kf_model_param["y_test"], model_gnb["yhat"])

		model_gnb["roc"] = roc_curve(dict_kf_model_param["y_test"], model_gnb["pop_polarity"], pos_label=1)
		model_gnb["auc"] = round(roc_auc_score(dict_kf_model_param["y_test"], model_gnb["pop_polarity"]), 2)

		print( "\n pop_polarity" )
		print( model_gnb["pop_polarity"] )

		print( "\n yhat" )
		print( model_gnb["yhat"] )

		print( "\n auc" )
		print( model_gnb["auc"] )


		"""Confusion Matrix"""

		print( """\n model_gnb["con_matrix"]""" )
		print( model_gnb["con_matrix"] )

		# True Positives
		model_gnb["TP"] = model_gnb["con_matrix"][1, 1]

		# True Negatives
		model_gnb["TN"] = model_gnb["con_matrix"][0, 0]

		# False Positives
		model_gnb["FP"] = model_gnb["con_matrix"][0, 1]

		# False Negatives
		model_gnb["FN"] = model_gnb["con_matrix"][1, 0]

		model_gnb["accuracy"] = (model_gnb["TP"] + model_gnb["TN"]) / float(model_gnb["TP"] + model_gnb["TN"] + model_gnb["FP"] + model_gnb["FN"])
		model_gnb["recall"] = model_gnb["TP"] / float(model_gnb["TP"] + model_gnb["FN"])
		model_gnb["specificity"] = model_gnb["TN"] / float(model_gnb["TN"] + model_gnb["FP"])
		model_gnb["precision"] = model_gnb["TP"] / float(model_gnb["TP"] + model_gnb["FP"])

		print("\n Model performance calculate from confusion matrix:\n" )
		print("accuracy: ", model_gnb["accuracy"] )
		print("recall: ",model_gnb["recall"] )
		print("specificity: ",model_gnb["specificity"] )
		print("precision: ",model_gnb["precision"] )

		print ( """\n model_gnb["class_report"]""" )
		print ( model_gnb["class_report"] )
		
		self.dict_kf_model_gnb[dict_kf_model_param["kf_id"]] = model_gnb
		
		
	def random_forest(self, dict_kf_model_param):
		#
		print ( "\n random_forest() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("nKFold round: ", dict_kf_model_param["kf_id"])
		#
		model_rf = {}
		model_rf = mml.model_ml(X = dict_kf_model_param["X_train"], y = dict_kf_model_param["y_train"]).random_forest()
		
		"""Evaluate and predict"""

		model_rf["pop_polarity"] = model_rf["tuned_model"].predict_proba(dict_kf_model_param["X_test"])[:, 1]
		model_rf["yhat"] = model_rf["tuned_model"].predict(dict_kf_model_param["X_test"])

		model_rf["con_matrix"] = confusion_matrix(dict_kf_model_param["y_test"], model_rf["yhat"], labels=[0, 1])
		model_rf["class_report"] = classification_report(dict_kf_model_param["y_test"], model_rf["yhat"])

		model_rf["roc"] = roc_curve(dict_kf_model_param["y_test"], model_rf["pop_polarity"], pos_label=1)
		model_rf["auc"] = round(roc_auc_score(dict_kf_model_param["y_test"], model_rf["pop_polarity"]), 2)

		print( "\n pop_polarity" )
		print( model_rf["pop_polarity"] )

		print( "\n yhat" )
		print( model_rf["yhat"] )

		print( "\n auc" )
		print( model_rf["auc"] )


		"""Confusion Matrix"""

		print( """\n model_rf["con_matrix"]""" )
		print( model_rf["con_matrix"] )

		# True Positives
		model_rf["TP"] = model_rf["con_matrix"][1, 1]

		# True Negatives
		model_rf["TN"] = model_rf["con_matrix"][0, 0]

		# False Positives
		model_rf["FP"] = model_rf["con_matrix"][0, 1]

		# False Negatives
		model_rf["FN"] = model_rf["con_matrix"][1, 0]

		model_rf["accuracy"] = (model_rf["TP"] + model_rf["TN"]) / float(model_rf["TP"] + model_rf["TN"] + model_rf["FP"] + model_rf["FN"])
		model_rf["recall"] = model_rf["TP"] / float(model_rf["TP"] + model_rf["FN"])
		model_rf["specificity"] = model_rf["TN"] / float(model_rf["TN"] + model_rf["FP"])
		model_rf["precision"] = model_rf["TP"] / float(model_rf["TP"] + model_rf["FP"])

		print("\n Model performance calculate from confusion matrix:\n" )
		print("accuracy: ", model_rf["accuracy"] )
		print("recall: ",model_rf["recall"] )
		print("specificity: ",model_rf["specificity"] )
		print("precision: ",model_rf["precision"] )

		print ( """\n model_rf["class_report"]""" )
		print ( model_rf["class_report"] )

		self.dict_kf_model_rf[dict_kf_model_param["kf_id"]] = model_rf


	def svm(self, dict_kf_model_param):
		#
		print ( "\n svm() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("nKFold round: ", dict_kf_model_param["kf_id"])
		#
		model_svm = {}
		model_svm = mml.model_ml(X = dict_kf_model_param["X_train"], y = dict_kf_model_param["y_train"]).svm()
		
		"""Evaluate and predict"""

		model_svm["pop_polarity"] = model_svm["tuned_model"].predict_proba(dict_kf_model_param["X_test"])[:, 1]
		model_svm["yhat"] = model_svm["tuned_model"].predict(dict_kf_model_param["X_test"])

		model_svm["con_matrix"] = confusion_matrix(dict_kf_model_param["y_test"], model_svm["yhat"], labels=[0, 1])
		model_svm["class_report"] = classification_report(dict_kf_model_param["y_test"], model_svm["yhat"])

		model_svm["roc"] = roc_curve(dict_kf_model_param["y_test"], model_svm["pop_polarity"], pos_label=1)
		model_svm["auc"] = round(roc_auc_score(dict_kf_model_param["y_test"], model_svm["pop_polarity"]), 2)

		print( "\n pop_polarity" )
		print( model_svm["pop_polarity"] )

		print( "\n yhat" )
		print( model_svm["yhat"] )

		print( "\n auc" )
		print( model_svm["auc"] )


		"""Confusion Matrix"""

		print( """\n model_svm["con_matrix"]""" )
		print( model_svm["con_matrix"] )

		# True Positives
		model_svm["TP"] = model_svm["con_matrix"][1, 1]

		# True Negatives
		model_svm["TN"] = model_svm["con_matrix"][0, 0]

		# False Positives
		model_svm["FP"] = model_svm["con_matrix"][0, 1]

		# False Negatives
		model_svm["FN"] = model_svm["con_matrix"][1, 0]

		model_svm["accuracy"] = (model_svm["TP"] + model_svm["TN"]) / float(model_svm["TP"] + model_svm["TN"] + model_svm["FP"] + model_svm["FN"])
		model_svm["recall"] = model_svm["TP"] / float(model_svm["TP"] + model_svm["FN"])
		model_svm["specificity"] = model_svm["TN"] / float(model_svm["TN"] + model_svm["FP"])
		model_svm["precision"] = model_svm["TP"] / float(model_svm["TP"] + model_svm["FP"])

		print("\n Model performance calculate from confusion matrix:\n" )
		print("accuracy: ", model_svm["accuracy"] )
		print("recall: ",model_svm["recall"] )
		print("specificity: ",model_svm["specificity"] )
		print("precision: ",model_svm["precision"] )

		print ( """\n model_svm["class_report"]""" )
		print ( model_svm["class_report"] )

		self.dict_kf_model_svm[dict_kf_model_param["kf_id"]] = model_svm


	def xgboost(self, dict_kf_model_param):
		#
		print ( "\n xgboost() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		print("nKFold round: ", dict_kf_model_param["kf_id"])
		#
		model_xgb = {}
		model_xgb = mml.model_ml(X = dict_kf_model_param["X_train"], y = dict_kf_model_param["y_train"]).xgboost()
		
		"""Evaluate and predict"""

		model_xgb["pop_polarity"] = model_xgb["tuned_model"].predict_proba(dict_kf_model_param["X_test"])[:, 1]
		model_xgb["yhat"] = model_xgb["tuned_model"].predict(dict_kf_model_param["X_test"])

		model_xgb["con_matrix"] = confusion_matrix(dict_kf_model_param["y_test"], model_xgb["yhat"], labels=[0, 1])
		model_xgb["class_report"] = classification_report(dict_kf_model_param["y_test"], model_xgb["yhat"])

		model_xgb["roc"] = roc_curve(dict_kf_model_param["y_test"], model_xgb["pop_polarity"], pos_label=1)
		model_xgb["auc"] = round(roc_auc_score(dict_kf_model_param["y_test"], model_xgb["pop_polarity"]), 2)

		print( "\n pop_polarity" )
		print( model_xgb["pop_polarity"] )

		print( "\n yhat" )
		print( model_xgb["yhat"] )

		print( "\n auc" )
		print( model_xgb["auc"] )


		"""Confusion Matrix"""

		print( """\n model_xgb["con_matrix"]""" )
		print( model_xgb["con_matrix"] )

		# True Positives
		model_xgb["TP"] = model_xgb["con_matrix"][1, 1]

		# True Negatives
		model_xgb["TN"] = model_xgb["con_matrix"][0, 0]

		# False Positives
		model_xgb["FP"] = model_xgb["con_matrix"][0, 1]

		# False Negatives
		model_xgb["FN"] = model_xgb["con_matrix"][1, 0]

		model_xgb["accuracy"] = (model_xgb["TP"] + model_xgb["TN"]) / float(model_xgb["TP"] + model_xgb["TN"] + model_xgb["FP"] + model_xgb["FN"])
		model_xgb["recall"] = model_xgb["TP"] / float(model_xgb["TP"] + model_xgb["FN"])
		model_xgb["specificity"] = model_xgb["TN"] / float(model_xgb["TN"] + model_xgb["FP"])
		model_xgb["precision"] = model_xgb["TP"] / float(model_xgb["TP"] + model_xgb["FP"])

		print("\n Model performance calculate from confusion matrix:\n" )
		print("accuracy: ", model_xgb["accuracy"] )
		print("recall: ",model_xgb["recall"] )
		print("specificity: ",model_xgb["specificity"] )
		print("precision: ",model_xgb["precision"] )

		print ( """\n model_xgb["class_report"]""" )
		print ( model_xgb["class_report"] )

		self.dict_kf_model_xgb[dict_kf_model_param["kf_id"]] = model_xgb
	

	def get_max_index(self, dict_model):
		#
		print ( "\n get_max_index() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#
		for i in range(self.k_fold):	
			if i == 0:					
				max_value = dict_model[i]["accuracy"]
				max_index = i					
				temp_value = dict_model[i]["accuracy"]
				temp_index = i
			else:
				temp_value = dict_model[i]["accuracy"]
				temp_index = i
				#
				if max_value < temp_value:
					max_value = temp_value
					max_index = temp_index
		#
		return i
		
		
	def compare_final_result(self):
		#
		print ( "\n compare_final_result() is activated...\n" )
		#
		print("\nHostpital: " + self.dict_param["hosp_name"])
		#		
		
		#Logistic baseline
		idx_lgr = self.get_max_index(self.dict_kf_model_lgr)		
		#		
		print( "\n===== ", "Logistic baseline" , " =====" )
		print( "accuracy = ", self.dict_kf_model_lgr[idx_lgr]["accuracy"] )
		print( "recall = ", self.dict_kf_model_lgr[idx_lgr]["recall"] )
		print( "specificity = ", self.dict_kf_model_lgr[idx_lgr]["specificity"] )
		print( "precision = ", self.dict_kf_model_lgr[idx_lgr]["precision"] )
		print( "auc = ", self.dict_kf_model_lgr[idx_lgr]["auc"] )
		
		
		#GaussianNB
		idx_gnb = self.get_max_index(self.dict_kf_model_gnb)		
		#		
		print( "\n===== ", "GaussianNB" , " =====" )
		print( "accuracy = ", self.dict_kf_model_gnb[idx_gnb]["accuracy"] )
		print( "recall = ", self.dict_kf_model_gnb[idx_gnb]["recall"] )
		print( "specificity = ", self.dict_kf_model_gnb[idx_gnb]["specificity"] )
		print( "precision = ", self.dict_kf_model_gnb[idx_gnb]["precision"] )
		print( "auc = ", self.dict_kf_model_gnb[idx_gnb]["auc"] )
		
		
		#Random forest
		idx_rf = self.get_max_index(self.dict_kf_model_rf)
		#				
		print( "\n===== ", "Random forest" , " =====" )
		print( "accuracy = ", self.dict_kf_model_rf[idx_rf]["accuracy"] )
		print( "recall = ", self.dict_kf_model_rf[idx_rf]["recall"] )
		print( "specificity = ", self.dict_kf_model_rf[idx_rf]["specificity"] )
		print( "precision = ", self.dict_kf_model_rf[idx_rf]["precision"] )
		print( "auc = ", self.dict_kf_model_rf[idx_rf]["auc"] )
		
		
		#SVM
		idx_svm = self.get_max_index(self.dict_kf_model_svm)
		#	
		print( "\n===== ", "SVM" , " =====" )
		print( "accuracy = ", self.dict_kf_model_svm[idx_svm]["accuracy"] )
		print( "recall = ", self.dict_kf_model_svm[idx_svm]["recall"] )
		print( "specificity = ", self.dict_kf_model_svm[idx_svm]["specificity"] )
		print( "precision = ", self.dict_kf_model_svm[idx_svm]["precision"] )
		print( "auc = ", self.dict_kf_model_svm[idx_svm]["auc"] )
		
		
		#XGBoost
		idx_xgb = self.get_max_index(self.dict_kf_model_xgb)
		#		
		print( "\n===== ", "XGBoost" , " =====" )
		print( "accuracy = ", self.dict_kf_model_xgb[idx_xgb]["accuracy"] )
		print( "recall = ", self.dict_kf_model_xgb[idx_xgb]["recall"] )
		print( "specificity = ", self.dict_kf_model_xgb[idx_xgb]["specificity"] )
		print( "precision = ", self.dict_kf_model_xgb[idx_xgb]["precision"] )
		print( "auc = ", self.dict_kf_model_xgb[idx_xgb]["auc"] )
