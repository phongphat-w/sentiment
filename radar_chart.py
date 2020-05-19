# -*- coding: utf-8 -*-
"""
@author: Phongphat Wiwatthanasetthakarn
@create: 2020-04-07
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class radar_chart():

	def __init__(self, chart_title, df_input, list_label, dict_color, dict_series_name, export_file_name):
		
		self.chart_title = chart_title
		self.df_input = df_input
		self.list_label = list_label		
		self.dict_color = dict_color
		self.dict_series_name = dict_series_name
		self.export_file_name = export_file_name
	
	
	def radar_plot(self):
		# 
		print ( "\n radar_plot() is activated...\n" )
		#
		get_ipython().run_line_magic('matplotlib', 'inline')
		#
		# Each attribute we'll plot in the radar chart.
		labels = self.list_label
		#print("\nlabels: ", labels)

		# Number of variables we're plotting.
		num_vars = len(labels)

		# Split the circle into even parts and save the angles
		# so we know where to put each axis.
		angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

		# The plot is a circle, so we need to "complete the loop"
		# and append the start value to the end.
		angles += angles[:1]

		# ax = plt.subplot(polar=True)
		fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
				
		#fig = plt.figure()
		#ax = fig.add_subplot(111, projection="polar")
		

		for i in range(len(self.df_input)):	#loop for each hospital
			#
			#values = self.df_input.loc["total_review"].tolist()
			values = [self.df_input.loc[i,"doctor"], 
				self.df_input.loc[i,"nurse"],
				self.df_input.loc[i,"patient"],
				self.df_input.loc[i,"service"],
				self.df_input.loc[i,"time"]]
			#
			values += values[:1]
			
			print ("\nvalues: ", values)
			
			ax.plot(angles, values, color=self.dict_color[i], linewidth=1, label=self.dict_series_name[i])
			ax.fill(angles, values, color=self.dict_color[i], alpha=0.25)
			#plt.show()
		
		# Fix axis to go in the right order and start at 12 o'clock.
		ax.set_theta_offset(np.pi / 2)
		ax.set_theta_direction(-1)

		# Draw axis lines for each angle and label.
		ax.set_thetagrids(np.degrees(angles), labels)

		# Go through labels and adjust alignment based on where
		# it is in the circle.
		for label, angle in zip(ax.get_xticklabels(), angles):
			if angle in (0, np.pi):
				label.set_horizontalalignment('center')
			elif 0 < angle < np.pi:
				label.set_horizontalalignment('left')
			else:
				label.set_horizontalalignment('right')

		# Ensure radar goes from 0 to 100.
		ax.set_ylim(0, 1)
		# You can also set gridlines manually like this:
		# ax.set_rgrids([20, 40, 60, 80, 100])

		# Set position of y-labels (0-100) to be in the middle
		# of the first two axes.
		ax.set_rlabel_position(180 / num_vars)

		# Add some custom styling.
		# Change the color of the tick labels.
		ax.tick_params(colors='#222222')
		# Make the y-axis (0-100) labels smaller.
		ax.tick_params(axis='y', labelsize=8) #8
		# Change the color of the circular gridlines.
		ax.grid(color='#AAAAAA')
		# Change the color of the outermost gridline (the spine).
		ax.spines['polar'].set_color('#222222')
		# Change the background color inside the circle itself.
		ax.set_facecolor('#FAFAFA')

		# Add title.
		ax.set_title(self.chart_title, y=1.08)

		# Add a legend as well.
		ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
		
		plt.show()
		#
