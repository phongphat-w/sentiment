# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
@create: 2020-04-07
"""

#Utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import resample

class data_loader():

	def __init__(self, hosp_name=None):
		#
		self.scraping_path = "./scraping/"
		#
		self.hosp_name = hosp_name
		#
		self.df_nlp = pd.DataFrame()
		self.df_nlp_up = pd.DataFrame()
		#
		self.df_nlp_pos = pd.DataFrame()
		self.df_nlp_neg = pd.DataFrame()
		#
		self.load_dataset()
		

	def load_dataset(self):   
		#Read data for NLP process   
		#
		print ( "\n load_dataset() is activated...\n" )
		
		#		
		self.df_nlp = pd.read_csv( self.scraping_path + "comment_" + self.hosp_name + "_en.csv", usecols = ['score','en'] )

		# filtering to exclude score = 3 (Neutral)
		self.df_nlp = self.df_nlp[ self.df_nlp["score"] != 3  ]
		self.df_nlp["polarity"] = self.df_nlp.progress_apply( lambda x: 1 if x["score"] > 3 else 0  , axis=1 )

		print ( "\n>> Positive and negative dataset:" )
		print(self.df_nlp)

		self.df_nlp_pos = self.df_nlp[ self.df_nlp["polarity"] == 1  ]
		self.df_nlp_neg = self.df_nlp[ self.df_nlp["polarity"] == 0  ]

		print ( "\n>> Positive dataset:" )
		print(self.df_nlp_pos)

		print ( "\n>> Negative dataset:" )
		print(self.df_nlp_neg)


	def plot_data(self, df):   
		#Plot dataset group by polarity
		#
		print ( "\n plot_data() is activated...\n" )
		# 
		#%matplotlib inline
		#
		ax = df["polarity"].value_counts(sort=False).plot(kind='barh', legend = False, title = self.hosp_name)
		ax.set_xlabel("Number of Samples")
		ax.set_ylabel("0=Nagative, 1=Positive") 
		plt.show()

		#Summary  

		#df['polarity'].value_counts()

		record_total = len(df.index)        
		record_pos = len(     df[ df["polarity"] == 1  ]     )
		record_neg = len(     df[ df["polarity"] == 0  ]     )

		print ( "\nTotal = {}".format(record_total) )
		print ( "Positive = {}".format(record_pos) )
		print ( "Nagative = {}".format(record_neg) )

		print ( "\nDataframe size")
		print (df.shape)


	def up_sample(self):
		#Generate sample for minority class
		#
		print ( "\n up_sample() is activated...\n" )
		#print ("\nHospital name: " + self.hosp_name)
		# 
		df_nlp_major = self.df_nlp[ self.df_nlp.polarity == 1 ]
		df_nlp_minor = self.df_nlp[ self.df_nlp.polarity == 0 ]

		num_major_sample_size = len(df_nlp_major)

		df_minor_upsample = resample(df_nlp_minor, 
									 replace = True,     
									 n_samples = num_major_sample_size,    
									 random_state = 123)

		self.df_nlp_up = pd.concat( [df_nlp_major, df_minor_upsample] )
		self.df_nlp_up.reset_index(inplace = True, drop=True) 
		
		self.df_nlp_up = self.df_nlp_up.sample(frac=1)
		self.df_nlp_up.reset_index(inplace = True, drop=True)

		print ( "\nDataset after up sample\n")
		print( self.df_nlp_up )

		print ( "\nDataset after up sample group by class\n")
		print( self.df_nlp_up["polarity"].value_counts() )
		
		self.plot_data(self.df_nlp_up)
		