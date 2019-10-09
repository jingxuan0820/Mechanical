import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def main(filter_time):
	data = pd.read_csv('Input/Accelerometer.txt', header=1, sep='\t', dtype='float64')
	data_in_RAM = data.values
	n_data = data_in_RAM.shape[0]
	initial_time = 0
	step_time = filter_time
	# Data is considered in order

	for i in range(n_data):
		temp_list = []
		if data.iloc[i,0] > step_time:

			step_time += filter_time



if __name__ == '__main__':
	filter_time = 1 # This value is in seconds
	main(filter_time)
