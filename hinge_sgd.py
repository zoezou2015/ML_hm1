# Author: Zou Yanyan
#
# Function: Implement stochastic gradient descent algorithm that 
#           minimize hinge loss to tackle binary classification problem

from __future__ import division
import numpy 
from numpy import *
from random import shuffle
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt


import document_vectorize
#import evaluation
import predict




category = ['atheism', 'sports']
path1 = '/Users/Zoe/Desktop/HW1/data/train'
path2 = '/Users/Zoe/Desktop/HW1/data/test'



def hinge_sgd(train_path, test_path, category, k):
	"""
	k refers to the category whose classified label is set 1
	'0' : 'atheism'
	'1' : 'politics'
	'2' : 'science'
	'3' : 'sports'

	"""
	print '------------Stochastic gradient descent with hinge loss------------'
	print 'Loading data...'

	#load data
	[x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category, k)

	MaxIteration = 200
	
	sample_num = train_size[0]
	feature_num = train_size[1]

	

	# start training 
	print 'Start training...'
	

	learn_rates = [  0.001, 0.01, 0.1, 1, 10, 20, 50, 100, 500]
	

	for learn_rate in learn_rates:
		print 'learn_rate: ',learn_rate	
		#if min_loss == 0:
		#	break		
		min_loss = 10000000

		update_loss = 0
		shuffle_order = range(sample_num)
	

		
		w = zeros(feature_num)
		b = 0
		min_w = zeros(feature_num)	
		min_b = 0
		start_time = time.time()
		for iteration in range(MaxIteration):
			
			#learn_rate = 1/(iteration+1)
				
				#stochastic gradient descent
				
			shuffle(shuffle_order) 

			for t in shuffle_order:
				#if point == random_point_size - 1:
					#print "*"
					#t = random.randint(0, sample_num-1)
				if ((numpy.inner(x[t].toarray(),w)+b))*y[t] <= 0:
	 				w = numpy.add(w, numpy.multiply(x[t].toarray(), learn_rate * y[t]))
					b += learn_rate * y[t]

				#calculate hinge loss
			temp_loss = 0
			for i in range (sample_num):
				sample_loss = ((numpy.inner(x[i].toarray(),w)+b))*y[i]
				if sample_loss < 1:
					temp_loss += 1 - sample_loss
					update_loss = temp_loss/sample_num

			#print'u',update_loss
				# record minimum loss
			if min_loss > update_loss:
				min_loss = update_loss
				min_w = w
				min_b = b 
			if min_loss == 0:
				break
				#print "min_loss = "+str(min_loss)
				#print "iteration = "+str(iteration)

				#if abs(min_loss - update_loss) < 0.000001:
				#	break
				
		print 'Training time: ', time.time() - start_time
		print "RESULT: w: " + str(min_w) + " b: " + str(min_b)

		predict.predict(x, y, train_size, x_test,y_test,test_size, min_w, min_b)





#hinge_sgd(path1,path2,category,0)

#if __name__ == "__main__":
    #training_set = document_vectorize.createDataSet(path1,category1)
    
 #   pass















