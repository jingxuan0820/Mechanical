import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def main(filter_time):
	logger.info('Starting the preprocessing of data for simplificate the analysis, Its taken {} second like filter time'.format(filter_time))
	data = pd.read_csv('Input/Accelerometer.txt', header=1, sep='\t', dtype='float64')
	data_columns = data.columns
	n_data = data.shape[0]
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

		if data.iloc[i,0] <= step_time:
			acceleration_temp = acceleration_temp + data.iloc[i,1:]
			cont_acceleration_temp += 1
			time_temp += data.iloc[i,0]
		else:
			preprocessing_data = np.append(preprocessing_data, time_temp/cont_acceleration_temp)
			preprocessing_data = np.append(preprocessing_data, acceleration_temp/cont_acceleration_temp)
			step_time += filter_time
			acceleration_temp = data.iloc[i,1:].values
			cont_acceleration_temp = 0
			time_temp = 0.0
	preprocessing_data = preprocessing_data.reshape((int(len(preprocessing_data)/5),5))
	
	data_calculated = pd.DataFrame(preprocessing_data,columns = data_columns, dtype = 'float64')

	data_calculated.to_csv('output.csv', sep = '\t', index = False)


if __name__ == '__main__':
	filter_time = 1 # This value is in seconds
	main(filter_time)
