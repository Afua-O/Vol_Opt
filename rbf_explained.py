# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 12:50:31 2021

@author: aow001
"""
"""
J_env = f(downstream release, target release)
downstream release = f(available storage, prescribed policy (uu))
uu = f (RBF input)
RBF input = f(day of year, available storage)

Let  day of year/ time = day 1
available storage = 108 

"""

import numpy as np
from numpy import random

#starting parameters
time = 1 # day of year
reservoir_level = 108 # starting level of reservoir (historical mean)
n_rbfs = 4 # number of radial basis functions (RBF) 
n_inputs = 2 #number of inputs to the rbf functions. A function of the time and storage
n_outputs = n_rbfs

"""n_rbf: determined by the number of unique water users to which water is being 
allocated. For Sasqurhana, this is 4: the 3 consumptive uses (Baltimore, Chester, Atomic) 
and then the nonconsumptive uses, hydro, env and weekend levels, taken as one unique 
user as releases for any one of these 3 also helps to meet the other 2"""

inputs1 = np.array([time, reservoir_level])
inputs = inputs1[np.newaxis, :] #the np.newaxis function creates a new array, converting the 1D array into 
#a 2D array, thus making the subtraction possible

centers = np.full((n_rbfs, n_inputs), -0.5) #randomly assigned from the range [-1,1] (in this case all elements selected as 0.5) 
radii = np.full((n_rbfs, n_inputs), 0.9) #randomly assigned from the range [0,1] (in this case all elements selected as 0.1) 
weights = np.full((n_rbfs, n_outputs), 0.11) #randomly assigned from the range [0,1] (in this case all elements selected as 0.5) 


#Squared Exponential RBF
A= inputs-centers
B = A ** 2 #not matrix arithmetric but squaring of individual elements
C = radii ** 2
rbf_scores = np.exp(-(np.sum(B / C, axis=1))) #exp(-b/c) 
#NB not training set so RBF scores and eventual output will be 0 array

 # n_rbf x n_output, n_rbf
weighted_rbfs = weights * (rbf_scores[:, np.newaxis])
output = weighted_rbfs.sum(axis=0)

print("Input1 matrix size")
print(np.shape(inputs1))   #print shape of matrix
print(" Input1 matrix:")
print (inputs1) # printing input matrix

print("Input matrix size")
print(np.shape(inputs))   #print shape of matrix
print(" Input matrix:")
print (inputs) # printing input matrix

print ( "Centers matrix size")
print(np.shape(centers))
print("Centers matrix")
print(centers)

print ( "Radii matrix size")
print(np.shape(radii))
print("Radii matrix")
print(radii)

print ( "Weight matrix size")
print(np.shape(weights))
print("Weights matrix")
print(weights)

print ( "Matrix A size")
print(np.shape(A))
print("Matrix A")
print(A)

print ( "Matrix B size")
print(np.shape(B))
print("Matrix B")
print(B)

print ( "Matrix C size")
print(np.shape(C))
print("Matrix C")
print(C)

print ( "RBF scores size")
print(np.shape(rbf_scores))
print("RBF scores")
print(rbf_scores)

print ( "Weighted rbfs size")
print(np.shape(weighted_rbfs))
print("Weighted rbfs")
print(weighted_rbfs)

print ( "Output size")
print(np.shape(output))
print("Output")
print(output)





