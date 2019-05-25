# !/usr/local/lib/python2.7 python 
# -*- coding=utf-8 -*-  

# bridging the python 2 and python 3 gap
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os # saving files
import numpy as np # for matrix math
import matplotlib.pyplot as plt # plot and visualizing data 
import scipy.io as sio # open matlab ".mat" data files

def read_data(data_mat):
	""" 
	READ POSCAR data from matlab file ".mat"

	Args: 
		data : mat file 

	Returns: 
		data : 4D numpy array
		
	"""
	# open POSCAR data
	# in dictionary 
	MH = sio.loadmat(data_mat)


	# delete some extras informations
	del MH['__version__']                                      
	del MH['__globals__']                                      
	del MH['__header__']

	# obtain dictionary key
	MH_keys = list(MH.keys())                     
	MH_k = MH_keys[0]  
	
	# Convert data into numpy array 4D 
	data = MH[MH_k]


	return(data)

def extract_features(data):

	abc = []; H = []; M1 = []; M2= [];
	for i in range(len(data)):                                   
	                                                             
	   abc.append(data[i][0])                                
	   H.append(data[i][1])
	   M1.append(data[i][2])
	   M2.append(data[i][3])
	return(abc, H, M1, M2) 

def compute_max(data):

	"""
	Compute the maximum length of features
	Args:
		data: First POSCAR data 

	Return:
		max_dim: Maximum length of the features in the 4D tensor
	"""

	abc, H, M1, M2 = extract_features(data)
	# list of length for each feature
	lens1 = np.array([len(H[i]) for i in range(len(H))])   
	lens2 = np.array([len(M1[i]) for i in range(len(M1))])
	lens3 = np.array([len(M2[i]) for i in range(len(M2))])    
	l1 = max(lens1)  
	l2 = max(lens2)  
	l3 = max(lens3)
	
	max_dim = max(l1,l2,l3)  

	return(max_dim)

def data_padding(data,l):

	""" 
	Padding data  to get the same dimension (4D) for each of both classes (data) included as input
	NOTE that CrystalGAN take as input two datasets:
	M1_H "Metal1 - Hydrogen" and M2_H "Metal2 - Hydrogen"

	Args: 
		data: POSCAR data for binary chemical componant in dictinary
		l : maximum length of features 

	Return:
		MH_data: POSCAR data with fixed shape and dimension 
	"""

	abc, H, M1, M2 = extract_features(data)
	ft1 = []; ft2 = []; ft3 = []; ft4 = []
	
	for i in range(len(abc)):                

	   zeros1 = np.zeros(3*l).reshape(l,3)                         
	   zeros2 = np.zeros(3*l).reshape(l,3)
	   zeros3 = np.zeros(3*l).reshape(l,3) 
	   zeros4 = np.zeros(3*l).reshape(l,3)
	   
	   zeros1[:abc[i].shape[0], :abc[i].shape[1]] = abc[i]    
	   zeros2[:H[i].shape[0], :H[i].shape[1]] = H[i]   
	   zeros3[:M1[i].shape[0], :M1[i].shape[1]] = M1[i] 
	   zeros4[:M2[i].shape[0], :M2[i].shape[1]] = M2[i] 


	   ft1.append(zeros1)                                                   
	   ft2.append(zeros2) 
	   ft3.append(zeros3) 
	   ft4.append(zeros4)
	                                                     
	MH_data = list(zip(ft1,ft2,ft3,ft4))

	return(MH_data)



