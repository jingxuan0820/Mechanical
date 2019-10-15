import numpy as np
import pandas as pd



class DataProcessor():

	def __init__(self, DataFrame):
		self.data = DataFrame


	def deltaAdjust(self, initial_acceleration, col_for_adj):
		columns = self.data.columns
		for j in col_for_adj:
			temp_vector = np.zeros((self.data.shape[0]))
			temp_vector[0] = initial_acceleration[j-1]
			for i in range(1,self.data.shape[0]):
				temp_vector[i] = self.data.iloc[i,j] - self.data.iloc[i-1,j]
			self.data.loc[:,columns[j]] = temp_vector

	def azAdjust(self,gravity):
		# this adjust is considering that the movil is not accelerated in z direction
		az_column = self.data.columns[4]
		temp_vector = np.zeros((self.data.shape[0]))
		for i in range(self.data.shape[0]):

			break
		return 0

	def adjustForFilterTime(self, filter_time):
		# logger is not imported
		#logger.info('Starting the preprocessing of data for simplificate the analysis, Its taken {} second like filter time'.format(filter_time))
		data = self.data
		data_columns = self.data.columns
		n_data = self.data.shape[0]
		initial_time = 0
		step_time = filter_time
		# Data is considered in order
		preprocessing_data = np.array([])
		preprocessing_data = np.append(preprocessing_data, data.iloc[0])

		acceleration_temp = np.array([0,0,0,0], dtype = 'float64')
		cont_acceleration_temp = 0
		time_temp = 0.0
		"""
		a = 0
		ax = 0
		ay = 0
		az = 0
		cont_a = 0
		cont_ax = 0
		cont_ay = 0
		cont_az = 0
		"""
		
		for i in range(1,n_data):

			if self.data.iloc[i,0] <= step_time:
				acceleration_temp = acceleration_temp + self.data.iloc[i,1:]
				cont_acceleration_temp += 1
				time_temp += self.data.iloc[i,0]
			else:
				preprocessing_data = np.append(preprocessing_data, time_temp/cont_acceleration_temp)
				preprocessing_data = np.append(preprocessing_data, acceleration_temp/cont_acceleration_temp)
				step_time += filter_time
				acceleration_temp = self.data.iloc[i,1:].values
				cont_acceleration_temp = 0
				time_temp = 0.0
		preprocessing_data = preprocessing_data.reshape((int(len(preprocessing_data)/5),5))
		
		self.data = pd.DataFrame(preprocessing_data,columns = data_columns, dtype = 'float64')

		self.data.to_csv('output.csv', sep = '\t', index = False)


	def recalculateAcceleration(self):
		pass

	def smoothData(self):
		pass

	def calculateGravity(self):
		gravity = 0
		for i in range(self.data.shape[0]):

			gravity += self.data.iloc[i,1]
		return gravity/self.data.shape[0]





