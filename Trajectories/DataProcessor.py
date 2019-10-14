import numpy as np
import pandas as pd



class DataProcessor():

	def __init__(self, DataFrame):
		self.data = DataFrame


	def phyweAcceleration(self, initial_acceleration):
		columns = self.data.columns
		for j in range(1, self.data.shape[1]):
			temp_vector = np.zeros((self.data.shape[0]))
			temp_vector[0] = initial_acceleration[j-1]
			for i in range(1,self.data.shape[0]):
				temp_vector[i] = self.data.iloc[i,j] - self.data.iloc[i-1,j]
			self.data.loc[:,columns[j]] = temp_vector

	def 





