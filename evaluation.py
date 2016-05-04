from __future__ import division

def evaluation_binary(predict_cat, label_y):
    '''

    calculate predict accuracy

    '''
 
    correct = 0
    miss = 0
    total = len(predict_cat)

    for i in range(len(predict_cat)):
        if predict_cat[i] == label_y[i]:
            correct += 1
        else:
            miss += 1
   
    accuracy = correct/total 
      
    return accuracy
    


###--------------------DEBUG STATEMENTS----------------------
#print (correct + miss )== total
###--------------------DEBUG STATEMENTS----------------------

