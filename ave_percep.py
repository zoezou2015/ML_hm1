# Author: Zou Yanyan
#
# Function: Implement perceptron algorithms to perform
#           binary classification 
#

from __future__ import division
import numpy 
from numpy import *
from random import choice
from numpy import array, dot, random
import sklearn.metrics as metrics
import numpy.random 
import time
import matplotlib.pyplot as plt


import document_vectorize
#import evaluation
import predict
#category1 = ['atheism', 'sports']
#category2 = ['atheism','politics','science','sports']
#train_path = '/Users/Zoe/Desktop/HW1/data/train'
#test_path = '/Users/Zoe/Desktop/HW1/data/test'

 
def ave_percep(train_path, test_path, category, k):
	"""
	k refers to the category whose classified label is set 1
	'0' : 'atheism'
	'1' : 'politics'
	'2' : 'science'
	'3' : 'sports'

	"""
	print '------------------Averaged Perceptron Algorithm----------------'
	print 'Loading data...'
	#load data
	[x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category,k)
	MaxIteration = 100
	sample_num = train_size[0]
	feature_num = train_size[1]

	w = zeros(feature_num)
	b = 0
	w_a = zeros(feature_num)
	b_a = 0
	c = 1
	w_ave = zeros(feature_num)
	b_ave = 0

	print 'Start training...'
	start_time = time.time()
	for iteration in range(MaxIteration):
		miss = 0
	#	print "iteration "+str(iteration)  
		for i in range(sample_num):
			#check misclassified point and modify weight
			if ((numpy.inner(x[i].toarray(),w)+b)) * y[i] <= 0:
				miss += 1
				w = numpy.add(w, numpy.multiply(x[i].toarray(),y[i]))
				w_a = numpy.add(w_a, numpy.multiply(x[i].toarray(),y[i]*c))
				b += y[i]
				b_a += y[i] * c
				c += 1
		#print "miss="+str(miss)

		w_ave = w - numpy.multiply(w_a, 1.0/c)
		b_ave = b - b_a/float(c)
		#print "miss:"+str(miss) 
		#print "iteration "+str(iteration) 
		if miss == 0:
			break
		#print "RESULT: w_ave: " + str(w_ave) + " b_ave: " + str(b_ave)

	print 'Training time: ', time.time() - start_time
	print "Result: w: " + str(w_ave) + " b: " + str(b_ave)
	print 'Start testing'
	predict.predict(x, y, train_size, x_test, y_test, test_size, w_ave, b_ave)


	return w_ave, b_ave

#ave_percep(train_path, test_path, category1, 0)



#if __name__ == "__main__":
    #training_set = document_vectorize.createDataSet(path1,category1)
    #    pass
	


