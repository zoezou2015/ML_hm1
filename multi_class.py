# Author: Zou Yanyan
#
# Function: Implement stochastic gradient descent algorithm that 
#           minimize hinge loss to tackle binary classification problem
#
from __future__ import division

import numpy as np
from numpy import *
import random
from numpy import array, dot, random

import document_vectorize
import evaluation
import hinge_regularization
import multi_perceptron

import predict

category1 = ['atheism', 'sports']
category2 = ['atheism','politics','science','sports']

path1 = '/Users/Zoe/Desktop/HW1/data/train'
path2 = '/Users/Zoe/Desktop/HW1/data/test'

def multi_classification(train_path, test_path, category):
	'''
	k refers to the category whose classified label is set 1
	'0' : 'atheism'
	'1' : 'politics'
	'2' : 'science'
	'3' : 'sports'

	'''

	total_categoray = 4
	print 'Call averaged perceptron algorithm to realize multi-class classification'
	for i in range(total_categoray):
		
		print '------------Classify ',category[i],'----------------'
		multi_perceptron.ave_percep(train_path,test_path,category,i)
#multi_classification(path1,path2,category2)
    

#if __name__ == "__main__":
    #training_set = document_vectorize.createDataSet(path1,category1)
    
    #pass














