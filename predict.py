from __future__ import division
import numpy 
from numpy import *
from numpy import array, dot, random
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt

import evaluation

def predict(x, y, train_size, x_test,y_test,test_size, w_ave, b_ave):
    """
    Predict training and test set

    """
    train_sample_num = train_size[0]
    train_feature_num = min(len(w_ave),train_size[1])
    train_predict_cat = zeros(train_sample_num)

    test_sample_num = test_size[0]
    test_feature_num = min(len(w_ave),test_size[1])
    test_predict_cat = zeros(test_sample_num)

    #predict training set
    for i in range(train_sample_num):
      if ((numpy.inner(x[i].toarray(),w_ave)+b_ave)) < 0:
          train_predict_cat[i] = -1
      else:
        train_predict_cat[i] = 1
    #print "# train_size = " + str(train_size)
    print 'Accuracy for training dataset: ',evaluation.evaluation_binary(train_predict_cat, y)

    #predict test set
    for i in range(test_sample_num):
      if ((numpy.inner(x_test[i].toarray(),w_ave)+b_ave)) < 0:
        test_predict_cat[i] = -1
      else:
        test_predict_cat[i] = 1
    #print "# test_size = " + str(test_size)
    print 'Accuracy for test dataset: ',evaluation.evaluation_binary(test_predict_cat, y_test)