
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
import predict
#import evaluation
#category1 = ['atheism', 'sports']
#category2 = ['atheism','politics','science','sports']
#train_path = '/Users/Zoe/Desktop/HW1/data/train'
#test_path = '/Users/Zoe/Desktop/HW1/data/test'


def perceptron(train_path, test_path, category, k):
  """
  k refers to the category whose classified label is set 1
  '0' : 'atheism'
  '1' : 'politics'
  '2' : 'science'
  '3' : 'sports'
    
  """
  print '--------------------- Perceptron Algorithm ----------------'
  print 'Loading data...'
  
  #load data
  [x, y, train_size,x_test,y_test,test_size] = document_vectorize.createDataSet(train_path, test_path, category,k)


  MaxIteration = 300
  sample_num = train_size[0]
  feature_num = train_size[1]
  w = zeros(feature_num)
  b = 0

  print 'Start training...'
  start_time = time.time()
  for iteration in range(MaxIteration):
    miss = 0

    for i in range(sample_num):
      #check misclassified point and modify weight        
      if ((numpy.inner(x[i].toarray(),w)+b))*y[i] <= 0:
        miss += 1
        w = numpy.add(w, numpy.multiply(x[i].toarray(),y[i]))
    #print "miss:"+str(miss) 
    #print "iteration "+str(iteration)  
    #print "RESULT: w: " + str(w) + " b: " + str(b) + " miss: " + str(miss)
    
    if miss == 0 :
      break
       
  print 'Training time: ', time.time() - start_time
  print "Result: w: " + str(w) + " b: " + str(b)
  print 'Start testing'
  predict.predict(x, y, train_size, x_test, y_test, test_size, w, b)
  

    

    


#perceptron(train_path, test_path, category1, 0)



#if __name__ == "__main__":
    #training_set = document_vectorize.createDataSet(path1,category1)
 #   c = 0
  #  while check(feature,label):
   #     c += 1
    #    pass

  

    
    
