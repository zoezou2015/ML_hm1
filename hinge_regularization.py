# Author: Zou Yanyan
#
# Function: Implement stochastic gradient descent algorithm that 
#           minimize hinge loss to tackle binary classification problem
#
from __future__ import division


import numpy 
from numpy import *
from random import shuffle
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt


import document_vectorize
import evaluation
import predict


#category = ['atheism', 'sports']
#path1 = '/Users/Zoe/Desktop/HW1/data/train'
#path2 = '/Users/Zoe/Desktop/HW1/data/test'


def hinge_regularization(train_path, test_path,category, k):
	"""
	k refers to the category whose classified label is set 1
	'0' : 'atheism'
	'1' : 'politics'
	'2' : 'science'
	'3' : 'sports'

	"""
	print '-----------Hinge loss with regularization term------------'
	print 'Loading data...'
	#load data

	[x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category,k)


	MaxIteration = 100
	
	
		
	sample_num = train_size[0]
	feature_num = train_size[1]

	
	lamdas = [0.0001, 0.001, 0.01, 0.1, 1, 10,100,1000,10000]

	
	
	for lamda in lamdas:
		w = zeros(feature_num)
		b = 0
		min_w = zeros(feature_num)  
		min_b = 0

		min_loss = 10000000
		update_loss = 0
		temp_loss = 0
		shuffle_order = range(sample_num)

		#start training
		print 'lamda: ', lamda
		print 'Start training...'
		start_time = time.time()


		for iteration in range(MaxIteration):
			#print 'itertion',iteration
			learn_rate = 1/(iteration + 1)

			shuffle(shuffle_order)
			#stochastic gradient descent
			for t in shuffle_order:
				#if point == random_point_size - 1:
				#   print '#'

				#t = random.randint(0, sample_num-1)
				if ((numpy.inner(x[t].toarray(),w)+b))*y[t] <= 0:
					#print "mm"
					w = numpy.add(w + numpy.multiply(x[t],lamda), numpy.multiply(x[t].toarray(), learn_rate * y[t]))
					b = learn_rate * y[t]

			#calculate loss function with regularization term
			temp_loss = 0
			for i in range (sample_num):
				sample_loss = ((numpy.inner(x[i].toarray(),w)+b))*y[i]
				if sample_loss < 1:
					temp_loss += 1 - sample_loss
				regular_term = numpy.inner(w,w)
			update_loss = temp_loss/sample_num + lamda * regular_term
			#print 'loss',update_loss
			#record minimun loss
			if min_loss > update_loss:
				min_loss = update_loss
				min_w = w
				min_b = b
			#print "iteration = "+str(iteration)
			#print "update_loss ="+str(update_loss)
			#print "min_loss =" +str(min_loss)


			

		#print 'Training time: ', time.time()-start_time
		#print 'lamda: ',lamda
		#print "RESULT: w: " + str(min_w) + " b: " + str(min_b)

		print 'Training time: ', time.time()-start_time
		print 'Result: w: ', min_w, 'b: ', min_b
		predict.predict(x, y, train_size, x_test, y_test, test_size, min_w, min_b)
		




#hinge_regularization(path1, path2, category, 0)




#if __name__ == "__main__":
	#hinge_regularization(path1, path2, category, 0)
 #   pass















