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

#---> pip install tqdm --upgrade
#---> pip install WordCloud
#---> pip install fonttools

import nltk
nltk.download("punkt")
#nltk.download("vader_lexicon")
nltk.download('stopwords')
nltk.download('wordnet')

#POS tag
nltk.download('averaged_perceptron_tagger')

#=====================

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re, string
#from nltk.corpus import twitter_samples
from nltk.corpus import stopwords 

#Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#Vector
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification
from collections import defaultdict
from nltk.text import TextCollection

from sklearn.decomposition import LatentDirichletAllocation as LDA

#Word cloud
from wordcloud import WordCloud

#Utilities
import numpy as np
import pandas as pd
from tqdm import tqdm

#SVG export
import fontTools
import fontTools.subset

#matplotlib
import matplotlib
#matplotlib.use('SVG') #set the backend to SVG
import matplotlib.pyplot as plt
import seaborn as sns
import radar_chart as radar

tqdm.pandas()


class nlp_process():

	def __init__(self, mode=None, hosp_name=None, dataset=None, dataset_type =None, description=None):
		#
		self.mode = mode
		#
		#Constant configuration
		self.mode_nlp = "nlp"
		self.mode_model = "model"
		#
		self.ngrams_unigrams = "unigrams"
		self.ngrams_bigrams = "bigrams"
		#
		self.word_cloud_path = "./wordcloud/"
		self.top_word_path = "./topword/"
		#
		self.hosp_name = hosp_name
		#
		#Prevent SettingWithCopyWarning from chained assignments
		#Protect original dataset from modification
		self.df_nlp = pd.DataFrame()
		self.df_nlp = dataset.copy()
		#
		self.dataset_type = dataset_type
		self.description = description
		#
		self.vec_tfidf_vectorizer = None
		self.data_tfidf_vectorizer = None
		#
		self.vec_tfidf_vectorizer_bigram = None
		self.data_tfidf_vectorizer_bigram = None
		#
		self.dict_count_freq_radar = {}
		self.dict_count_freq_word_cloud = {}
		self.dict_count_freq = {}
		
		self.counts_radar = []
		self.counts_word_cloud = []
		#
		#
		#
		print ("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print ("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
		print ("\n" + self.description)
		#
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Convert sentense to token <<<<<")
		self.gen_token()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Convert text to lowercase <<<<<")
		self.lowercase_text()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Remove special character <<<<<")
		self.remove_spec_char()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Remove stop word <<<<<")
		self.remove_stop_word()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Remove stop word (Special) <<<<<")
		print ("""\n>>>>> 've, ``, 's, n't, '', ' ' <<<<<""")
		self.remove_stop_word_spec()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Remove single and space token <<<<<")    
		self.remove_single_token()
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Normalization (Lemmatization: root word) <<<<<")    
		self.lemmatize_token()	
		
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Create vectors of Term Frequency–Inverse Document Frequency (TF-IDF) <<<<<")    
		self.vec_tf_idf()    

		if self.mode == self.mode_nlp:
			print ("\n----------------------------------------------------")
			print ("\n" + self.hosp_name)
			print ("\n" + self.description)
			print ("\n>>>>> Step: Create vectors of Latent Dirichlet Allocation (LDA) <<<<<")    
			self.vec_lda()

			
		print ("\n----------------------------------------------------")
		print ("\n" + self.hosp_name)
		print ("\n" + self.description)
		print ("\n>>>>> Step: Create vectors of Term Frequency–Inverse Document Frequency (TF-IDF), bigrams <<<<<")    
		self.vec_tf_idf_bigram()		
		
		if self.mode == self.mode_nlp:			
			print ("\n----------------------------------------------------")
			print ("\n" + self.hosp_name)
			print ("\n" + self.description)
			print ("\n>>>>> Step: Create vectors of Latent Dirichlet Allocation (LDA), bigrams <<<<<")    
			self.vec_lda_bigram()


	def gen_token(self):
		#Create token for sentense in corpus
		#
		print ( "\n gen_token() is activated...\n" )
		#		
		self.df_nlp["token"] = self.df_nlp.progress_apply( lambda x:  nltk.word_tokenize( x["en"]  ), axis=1 )
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)


	def lowercase_text(self):
		#Convert text to lowercase
		#
		print ( "\n lowercase_text() is activated...\n" )
		#
		self.df_nlp["lowercase"] = self.df_nlp.progress_apply( lambda x:  [word.lower() for word in x["token"] ] , axis=1 )
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)
        
        
	def remove_spec_char(self):
		#Remove special character
		#
		print ( "\n remove_spec_char() is activated...\n" )
		# 
		#REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
		REPLACE_NO_SPACE = re.compile("[.;:!\?,\"()\[\]]") #exclude " '  "
		REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
		#
		self.df_nlp["rem_spec_char"] = self.df_nlp.apply(lambda x: [REPLACE_NO_SPACE.sub( "", word ) for word in x["lowercase"] ], axis=1 )
		self.df_nlp["rem_spec_char"] = self.df_nlp.apply(lambda x: [REPLACE_WITH_SPACE.sub( " ", word ) for word in x["rem_spec_char"] ], axis=1 )
		#
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)
		

	def remove_stop_word(self):
		#Remove stop word
		#
		print ( "\n remove_stop_word() is activated...\n" )
		# 
		stop_words = set(stopwords.words('english'))
        #
		print ( "\nstop_words:\n" )
		print (stop_words)
		#
		self.df_nlp["stop_word"] = self.df_nlp.progress_apply(lambda x: [word for word in x["rem_spec_char"] if word not in stopwords.words()], axis=1 )
		#
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)


	def remove_stop_word_spec(self):
		#Remove stop word (Special) 
		#"""'ve""", """``""", """'s""", """n't""", """''""", """' '"""
		#
		print ( "\n remove_stop_word_spec() is activated...\n" )
		list_reserve = ["""'ve""", """``""", """'s""", """n't""", """''""", """' '""", """yet"""]
		#
		self.df_nlp["stop_word_02"] = self.df_nlp.progress_apply(lambda x: [word for word in x["stop_word"] if word not in list_reserve ], axis=1 )
		#
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)
		

	def remove_single_token(self):
		#Remove single and space token
		#
		print ( "\n remove_single_token() is activated...\n" )
		#
		self.df_nlp["rem_single_char"] = self.df_nlp.progress_apply(lambda x: [word for word in x["stop_word_02"] if (len(word) > 1)  ], axis=1 )
		#
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)


	def get_wordnet_pos(self, word):    
		#Map POS tag to first character lemmatize() accept
		#       
		tag = nltk.pos_tag([word])[0][1][0].upper()
		tag_dict = {"J": wordnet.ADJ,
					"N": wordnet.NOUN,
					"V": wordnet.VERB,
					"R": wordnet.ADV}

		return tag_dict.get(tag, wordnet.NOUN)


	def lemmatize_token(self):
		#Normalization (Lemmatization: root word)
		#
		print ( "\n lemmatize_token() is activated...\n" )
		# 
		obj_lemma = WordNetLemmatizer()
		#
		#Test POS
		print ( "\nTest POS:" )
		print ("\nThis is a book\n")
		print(nltk.pos_tag(nltk.word_tokenize("This is a book.")))
		#
		self.df_nlp["norm_lemma"] = self.df_nlp.progress_apply(lambda x: [obj_lemma.lemmatize(word, pos=self.get_wordnet_pos(word)  ) for word in x["rem_single_char"]  ], axis=1 )
		#
		print ("\nPreview some records:\n")
		#
		if self.df_nlp.empty != True:
			print (self.df_nlp)	
			print (self.df_nlp.shape)


	def vec_tf_idf(self):
		#Create vectors of Term Frequency–Inverse Document Frequency (TF-IDF)
		#
		print ( "\n vec_tf_idf() is activated...\n" )
		#
		self.df_nlp["de_token"] = self.df_nlp.progress_apply(lambda x: TreebankWordDetokenizer().detokenize(  x["norm_lemma"]  ), axis=1 )
		self.vec_tfidf_vectorizer = TfidfVectorizer()
		self.data_tfidf_vectorizer = self.vec_tfidf_vectorizer.fit_transform(self.df_nlp["de_token"])
		#
		print ( "\nPreview features name:" )
		print ( "\nNumber of features = {}".format( len(self.vec_tfidf_vectorizer.get_feature_names())) )
		print ( "\n\n" )
		print (self.vec_tfidf_vectorizer.get_feature_names())
		#
		print ( "\nPreview in matrix:\n" )
		print (self.data_tfidf_vectorizer.toarray())
		#
		print ( "\nPreview tf-idf score:\n" )
		print (self.data_tfidf_vectorizer)
		#		
		if self.mode == self.mode_nlp:
			word_cloud_file_name = "wc_" + self.hosp_name + "_" + self.dataset_type + ".png"
			top_word_file_name = "tw_" + self.hosp_name + "_" + self.dataset_type + ".png"
			#
			
			self.plot_top_most_words(self.data_tfidf_vectorizer, self.vec_tfidf_vectorizer, top_word_file_name, self.ngrams_unigrams)
			#self.plot_top_most_words(self.data_tfidf_vectorizer, self.vec_tfidf_vectorizer, top_word_file_name)
			#
			self.gen_word_cloud(word_cloud_file_name)
			

	def vec_tf_idf_bigram(self):
		#Create vectors of Term Frequency–Inverse Document Frequency (TF-IDF), bigrams
		#
		print ( "\n vec_tf_idf_bigram() is activated...\n" )
		#		
		self.vec_tfidf_vectorizer_bigram = TfidfVectorizer(ngram_range=(2, 2))
		self.data_tfidf_vectorizer_bigram = self.vec_tfidf_vectorizer_bigram.fit_transform(self.df_nlp["de_token"])
		#
		print ( "\nPreview features name:" )
		print ( "\nNumber of features = {}".format( len(self.vec_tfidf_vectorizer_bigram.get_feature_names())) )
		print ( "\n\n" )
		print (self.vec_tfidf_vectorizer_bigram.get_feature_names())
		#
		print ( "\nPreview in matrix:\n" )
		print (self.data_tfidf_vectorizer_bigram.toarray())
		#
		print ( "\nPreview tf-idf score:\n" )
		print (self.data_tfidf_vectorizer_bigram)
		#		
		if self.mode == self.mode_nlp:
			word_cloud_file_name = "wc_" + self.hosp_name + "_" + self.dataset_type + "_bigram.png"
			top_word_file_name = "tw_" + self.hosp_name + "_" + self.dataset_type + "_bigram.png"
			#	
			print ("\n\nTop most common words of " + self.hosp_name + """ (from TF-IDF vector, bigrams)""" )		
			self.plot_top_most_words(self.data_tfidf_vectorizer_bigram, self.vec_tfidf_vectorizer_bigram, top_word_file_name, self.ngrams_bigrams)
			#self.plot_top_most_words(self.data_tfidf_vectorizer_bigram, self.vec_tfidf_vectorizer_bigram, top_word_file_name)
			#
			self.gen_word_cloud(word_cloud_file_name)			


	def plot_top_most_words(self, count_data, count_vectorizer, export_file_name, ngram):
		#Plot top most common words
		# 
		get_ipython().run_line_magic('matplotlib', 'inline')
		#
		words = count_vectorizer.get_feature_names()
		#
		total_counts = np.zeros(len(words))
		for t in count_data:
			total_counts+=t.toarray()[0]
		#
		words = [(lambda x: x.replace(" ", "_"))(word) for word in words]        
        #
		#====================================
		#-----------------	
		words_wc = words
		total_counts_wc = total_counts		
		self.dict_count_freq_word_cloud = dict(zip(words_wc, total_counts_wc))
		
		#self.dict_count_freq = dict( zip(words, total_counts) ) 
		
		#-----------------
		self.dict_count_freq_radar = (zip(words, total_counts))
		#
		if ngram == self.ngrams_unigrams:
			list_include = ["service", "time","doctor", "nurse", "patient"]
			self.dict_count_freq_radar = ( (key,value) for key,value in self.dict_count_freq_radar if key in list_include )
			#self.dict_count_freq_radar = ( (key,value) for key,value in self.dict_count_freq_radar if key == "service" or key == "time" or key == "doctor"  or key ==  "nurse" or key ==  "patient" )
			
			#list_exclude = ["good", "rama","ramathibodi", "siriraj", "chulalongkorn", "bangkok"]
			#self.dict_count_freq_radar = ( (key,value) for key,value in self.dict_count_freq_radar if key not in list_exclude )
			
			#x[0] = name, x[1] = count value
			self.dict_count_freq_radar = sorted(self.dict_count_freq_radar, key=lambda x:x[0], reverse=False)[0:5]	
					
		else:
			#list_include = ["good_service"]
			#self.dict_count_freq_radar = ( (key,value) for key,value in self.dict_count_freq_radar if key in list_include )
			
			list_exclude = ["good_good", "rama_hospital", "ramathibodi_hospital", "siriraj_hospital", "chulalongkorn_hospital", "bangkok_hospital"]
			self.dict_count_freq_radar = ( (key,value) for key,value in self.dict_count_freq_radar if key not in list_exclude )
			#		
			self.dict_count_freq_radar = sorted(self.dict_count_freq_radar, key=lambda x:x[1], reverse=True)[0:10]
		#
		#self.dict_count_freq_word_cloud = sorted(self.dict_count_freq_word_cloud, key=lambda x:x[1], reverse=True)
		
		#print("\n\nself.dict_count_freq_word_cloud after sorted:", self.dict_count_freq_word_cloud)
		
		#
		words = [w[0] for w in self.dict_count_freq_radar]
		#words_wc = [w[0] for w in self.dict_count_freq_word_cloud]
		#
		##ORG: counts = [w[1] for w in self.dict_count_freq_word_cloud]
		self.counts_radar = [ 100 * (w[1] / len(count_vectorizer.get_feature_names()) ) for w in self.dict_count_freq_radar]
		#self.counts_word_cloud = [w[1] for w in self.dict_count_freq_word_cloud]
		
		#====================================
		x_pos = np.arange(len(words)) 
		#
		plt.figure(2, figsize=(12, 7))
		plt.subplot(title='10 most common words of ' + self.hosp_name)    
		sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
		sns_plot = sns.barplot(x_pos, self.counts_radar, palette='husl')
		plt.xticks(x_pos, words, rotation=20) 
		plt.xlabel('words')
		plt.ylabel('counts  (%)')
		#
		for p in sns_plot.patches:
			sns_plot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
		#
		plt.show()
		fig = sns_plot.get_figure()
		fig.savefig(self.top_word_path + export_file_name)
		#
		self.plot_radar_chart()


	def plot_radar_chart(self):
		#==================================
		#Radar chart
		#==================================
		print ("\n\nComment/review score(%) group by aspect of " + self.hosp_name + " (from TF-IDF vector)" )
		#
		#
		tup_input = list(self.dict_count_freq_radar)
		#
		df_input = pd.DataFrame()
		df_input.loc[0,"data_type"] = "total_review"
		df_input.loc[0,"doctor"] = self.counts_radar[0]
		df_input.loc[0,"nurse"] = self.counts_radar[1]
		df_input.loc[0,"patient"] = self.counts_radar[2]
		df_input.loc[0,"service"] = self.counts_radar[3]
		df_input.loc[0,"time"] = self.counts_radar[4]
		print("\n\n")
		print(df_input)
		#
		chart_title = "Review score(%) group by aspect of " + self.hosp_name
		#
		list_label = ["doctor", "nurse", "patient", "service", "time"]
		#
		dict_color = {}
		dict_color[0] =  "#1aaf6c"
		#
		dict_series_name = {}
		dict_series_name[0] =  self.hosp_name
		#
		export_file_name_radar = ""
		#
		obj_radar = radar.radar_chart(chart_title=chart_title, df_input=df_input, list_label=list_label, dict_color=dict_color, dict_series_name=dict_series_name, export_file_name=export_file_name_radar)
		obj_radar.radar_plot()
		

	def gen_word_cloud(self, export_file_name):
		#Generate word cloud
		#
		print ( "\n gen_word_cloud() is activated...\n" )
		# 
		get_ipython().run_line_magic('matplotlib', 'inline')
		#
		print ( "\n\nWord cloud for " + self.hosp_name )	
		print ( "\n" + self.description )		
		#
		wordcloud = WordCloud(width=700, height=400, background_color="white", max_words=3000, contour_width=3, contour_color='steelblue')
		wordcloud.generate_from_frequencies(self.dict_count_freq_word_cloud)
		#
		plt.figure(figsize=(18,8))
		plt_plot = plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis("off")
		plt.show()
		#
		fig = plt_plot.get_figure()
		fig.savefig(self.word_cloud_path + export_file_name)
		

	def print_topics(self, model, count_vectorizer, n_top_words):
		#print topics from LDA vectors
		words = count_vectorizer.get_feature_names()
		words = [(lambda x: x.replace(" ", "_"))(word) for word in words]
		#
		for topic_idx, topic in enumerate(model.components_):
			print("\nTopic #%d:" % topic_idx)
			print(", ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


	def vec_lda(self):
		#Create vectors of Latent Dirichlet Allocation (LDA)
		#
		print ( "\n vec_lda() is activated...\n" )
		# 
		number_topics = 5
		number_words = 10    
		#
		lda = LDA(n_components=number_topics, n_jobs=-1)
		lda.fit(self.data_tfidf_vectorizer)
		#
		print("Topics found by LDA:")
		self.print_topics(lda, self.vec_tfidf_vectorizer, number_words)


	def vec_lda_bigram(self):
		#Create vectors of Latent Dirichlet Allocation (LDA), bigrams
		#
		print ( "\n vec_lda_bigram() is activated...\n" )
		# 
		number_topics = 5
		number_words = 10
		#
		lda = LDA(n_components=number_topics, n_jobs=-1)
		lda.fit(self.data_tfidf_vectorizer_bigram)
		#
		print("Topics found by LDA (bigrams):")
		self.print_topics(lda, self.vec_tfidf_vectorizer_bigram, number_words)
		