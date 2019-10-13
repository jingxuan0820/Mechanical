import argparse
import logging
logging.basicConfig(level = logging.INFO)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

logger = logging.getLogger(__name__)

def main(filename, initial_velocity, initial_position):

	data = pd.read_csv(filename, header = 0, sep = '\t', dtype='float64')


	# Its taken the acceleration in cartesian coordinates, the first row is the time, and other three
	# are acceleration in the three components
	acceleration = data.iloc[:,2:].values
	time = data.iloc[:,0].values

	# It is possible to vectorize the next function for performance. For the moment it is not considered
	
	velocity_vector = np.array([antiderivative(time,acceleration[:,0],initial_velocity[0]),
							   antiderivative(time,acceleration[:,1],initial_velocity[1]),
							   antiderivative(time,acceleration[:,2],initial_velocity[2])]
							   )

	positional_vector = np.array([antiderivative(time,velocity_vector[0,:],initial_position[0]),
								 antiderivative(time,velocity_vector[1,:],initial_position[1]),
								 antiderivative(time,velocity_vector[2,:],initial_position[2]),]
								 )

	third_derivative = np.array([derivative(time,acceleration[:,0]),
								derivative(time,acceleration[:,1]),
								derivative(time,acceleration[:,2])]
								)
	
	velocity_vector = np.transpose(velocity_vector)
	positional_vector = np.transpose(positional_vector)
	third_derivative = np.transpose(third_derivative)

	# This three results are in vertical position, and is possible to add to DataFrame

	curvature = curvature_function(velocity_vector, acceleration)
	torsion = torsion_function(velocity_vector, acceleration, third_derivative)
	t_vector = tangent_vector_function(velocity_vector)
	n_vector = normal_vector_function(velocity_vector,acceleration)
	n_vector = normalize_vector(n_vector,t_vector, dimenssion = '2D')
	# This function allow us to guarantee n_vector is orthogonal to t_vector
	
	b_vector = binormal_vector_function(t_vector,n_vector)

	
	a_in_trajectory = acceleration_in_trajectory_function(t_vector,n_vector,b_vector,acceleration)
	

	logger.info('Starting to pass the data to csv for visualizing')

	name_columns_file1 = ['t in s','r_x (m)', 'r_y (m)', 'r_z (m)', 'v_x (m/s)','v_y (m/s)','v_z (m/s)',
						  'a_x (m/s2)', 'a_y (m/s2)', 'a_z (m/s2)', 'der(3)(r_x) (m/s3)',
						  'der(3)(r_y) (m/s3)','der(3)(r_z) (m/s3)']

	name_columns_file2 =['t in s','curvature (1/m)', 'torsion (m2)', 't_vector_x','t_vector_y',
						 't_vector_z','n_vector_x', 'n_vector_y','n_vector_z','b_vector_x',
						 'b_vector_y','b_vector_z', 'tangent acc (m/s2)','normal acc (m/s2)',
						 'binormal acc (m/s2)']

	dict_results_1 = { name_columns_file1[0]: time, name_columns_file1[1]: positional_vector[:,0],
					   name_columns_file1[2]: positional_vector[:,1], name_columns_file1[3]: positional_vector[:,2],
					   name_columns_file1[4]: velocity_vector[:,0], name_columns_file1[5]: velocity_vector[:,1],
					   name_columns_file1[6]: velocity_vector[:,2], name_columns_file1[7]: acceleration[:,0],
					   name_columns_file1[8]: acceleration[:,1],name_columns_file1[9]: acceleration[:,2],
					   name_columns_file1[10]: third_derivative[:,0], name_columns_file1[11]: third_derivative[:,1],
					   name_columns_file1[12]: third_derivative[:,2]
					}

	dict_results_2 = {name_columns_file2[0]: time, name_columns_file2[1]: curvature, name_columns_file2[2]: torsion,
					  name_columns_file2[3]: t_vector[:,0], name_columns_file2[4]: t_vector[:,1],
					  name_columns_file2[5]: t_vector[:,2], name_columns_file2[6]: n_vector[:,0],
					  name_columns_file2[7]: n_vector[:,1], name_columns_file2[8]: n_vector[:,2],
					  name_columns_file2[9]: b_vector[:,0], name_columns_file2[9]: b_vector[:,1],
					  name_columns_file2[9]: b_vector[:,2], name_columns_file2[10]: a_in_trajectory[:,0],
					  name_columns_file2[11]: a_in_trajectory[:,1], name_columns_file2[12]: a_in_trajectory[:,2]
					}

	cartesian_results = pd.DataFrame(dict_results_1)
	trajectory_results = pd.DataFrame(dict_results_2)
	cartesian_results.to_csv('cartesian_results.csv', sep = ';', index = False)
	trajectory_results.to_csv('trajectory_results.csv', sep = ';', index = False)

	logger.info('Starting the visualizing of data')

	fig = plt.figure()
	ax = plt.axes(projection="3d")

	ax.plot3D(list(positional_vector[:,0]),list(positional_vector[:,1]),list(positional_vector[:,2]), 'gray')

	plt.show()

def curvature_function(velocity_vector, acceleration):
	length = len(velocity_vector)
	curvature = np.zeros((length))
	if is_all_in_order(velocity_vector,acceleration):
		
		for i in range(length):
			curv = norm_of_vector(np.cross(velocity_vector[i],acceleration[i]))
			den_curv = norm_of_vector(velocity_vector[i])**3

			curvature[i] = curv/den_curv
		return curvature

def torsion_function(velocity_vector, acceleration, third_derivative):

	length = len(velocity_vector)
	torsion = np.zeros((length))
	for i in range(length):
		v1 = np.dot(velocity_vector[i],(np.cross(acceleration[i], third_derivative[i])))
		v2 = norm_of_vector(np.cross(velocity_vector[i],acceleration[i]))**2
		torsion[i] = v1/v2
	return torsion

def tangent_vector_function(velocity_vector):
	
	length = len(velocity_vector)
	t_vector = np.zeros((length,3))
	for i in range(length):
		t_vector[i] = velocity_vector[i] / norm_of_vector(velocity_vector[i])

	return t_vector

def normal_vector_function(velocity_vector,acceleration):

	length = len(velocity_vector)
	n_vector = np.zeros((length,3))
	for i in range(length):
		v1 = np.cross(velocity_vector[i],acceleration[i])
		v1 = np.cross(v1,velocity_vector[i])
		n_vector[i] = v1/ norm_of_vector(v1)
	return n_vector

def normalize_vector(n_vector,t_vector, dimenssion = '2D'):
	return n_vector

def binormal_vector_function(t_vector,n_vector):
	length = len(t_vector)
	binormal = np.zeros((length,3))
	for i in range(length):
		binormal[i] = np.cross(t_vector[i], n_vector[i])
	return binormal

def acceleration_in_trajectory_function(t_vector,n_vector,b_vector,acceleration):
	length = len(t_vector)
	a_in_trajectory = np.zeros((length,3))
	for i in range(length):
		tangential_component = np.dot(acceleration[i],t_vector[i])
		normal_component = np.dot(acceleration[i],n_vector[i])
		binormal_component = np.dot(acceleration[i],b_vector[i])
		a_in_trajectory[i] = np.array([tangential_component, normal_component, binormal_component])

	return a_in_trajectory


def norm_of_vector(vector):
	return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

def antiderivative(time, ordinate, initial_value, method = 0):
	# time and ordinate are vectors with the same dimenssion

	if method == 0:
		
		# Method of trapeze
		if is_all_in_order(time, ordinate):
			length = len(ordinate)
			result = np.zeros((length))
			result[0] = 0
			integral = 0
			for j in range(1,length):
				integral += (ordinate[j] + ordinate[j-1])*(time[j]-time[j-1])/2
				result[j] = integral
			result = result + initial_value
		return result

def derivative(time, ordinate, method = 0):
	# time and ordinate are vectors with the same dimenssion

	if method == 0:
		# simple method to calculate the derivative
		if is_all_in_order(time,ordinate):
			length = len(ordinate)
			result = np.zeros((length))

			v = (ordinate[1]-ordinate[0])/(time[1]-time[0])

			result[0] = v

			for i in range(1, length):
				derivative = (ordinate[i]-ordinate[i-1])/(time[i]-time[i-1])
				result[i] = derivative
		return result

def is_all_in_order(v1,v2):
	return True



if __name__ =='__main__':
	initial_velocity = np.array([0.0,0.0,0.0])
	initial_position = np.array([0.0,0.0,0.0])
	parser = argparse.ArgumentParser()
	parser.add_argument('filename',
						help = 'File to calculate Frenet Serret, data in cartesian',
						type = str)
	args = parser.parse_args()
	main(args.filename, initial_velocity, initial_position)
