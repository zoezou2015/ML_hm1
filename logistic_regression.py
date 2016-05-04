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






def logistic_reg(train_path, test_path ,category, k):
    """
    k refers to the category whose classified label is set 1
    '0' : 'atheism'
    '1' : 'politics'
    '2' : 'science'
    '3' : 'sports'
    
    """
    print '--------------------- Logistic Regression ----------------'
    print 'Loading data...'
    #load data
    [x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category, k)

    MaxIteration = 100
    min_loss = 10000000
    

    update_loss = 0
    sample_num = train_size[0]
    feature_num = train_size[1]

    w = zeros(feature_num)
    b = 0
    min_w = zeros(feature_num)  
    min_b = 0
    error = 0
   

    shuffle_order = range(sample_num)


    #start traing
    print 'Start traing...'
    start_time = time.time()
    for iteration in range(MaxIteration):

        learn_rate = 1/(iteration+1)
        shuffle(shuffle_order)

        #stochastic gradient descent
        for t in shuffle_order:
            #if point == random_point_size-1:
                #print '#'
            #t = random.randint(0, sample_num-1)
            
            temp1 = numpy.add(numpy.inner(w, x[t].toarray()), b)
            temp2 = numpy.exp(numpy.multiply(temp1, y[t]))
            temp3 = learn_rate / (1 + temp2)
       
            w = numpy.add(w, numpy.multiply(y[t] * temp3, x[t].toarray()))
            b = b + y[t] * temp3

        #print "iteration = "+str(iteration)
        #print "update_loss ="+str(update_loss)
        #print "min_loss =" +str(min_loss)

        #calculate loss
        temp_loss = 0
        for i in range (sample_num):
            temp1 = numpy.add(numpy.inner(w, x[t].toarray()), b)
            temp2 = numpy.exp(numpy.multiply(temp1, -y[t]))
            temp3 = 1 / (1 + temp2)
            temp_loss += numpy.log(1 + temp3)
        update_loss = temp_loss/sample_num

        #if min_loss == 0:
        #   break
        #record minimum loss
        if min_loss > update_loss:
            min_loss = update_loss
            min_w = w
            min_b = b

        #print "min_loss = "+str(min_loss)
        #print "iteration = "+str(iteration)
        #if abs(min_loss - update_loss) < 0.000001:
        #   break
        
    print 'Training time: ', time.time()-start_time
    print "RESULT: w: " + str(min_w) + " b: " + str(min_b)

    predict.predict(x, y, train_size, x_test, y_test, test_size, min_w, min_b)





#logistic_reg(path1,path2,category,0)

#if __name__ == "__main__":
    #training_set = document_vectorize.createDataSet(path1,category1)
    
    #pass















