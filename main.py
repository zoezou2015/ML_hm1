import perceptron1 
import ave_percep
import hinge_sgd
import hinge_regularization 
import logistic_regression 
import multi_class
import multi_perceptron




def main():

	category1 = ['atheism', 'sports']
	category2 = ['atheism','politics','science','sports']
	train_path = '/Users/Zoe/Documents/phd/Semester Jan-Apr 2015/Machine Learning/hw1/data/train'
	test_path = '/Users/Zoe/Documents/phd/Semester Jan-Apr 2015/Machine Learning/hw1/data/test'
	
	print '***Question 2: Compare perceptron and averaged perceptron***'
	#perceptron1.perceptron(train_path, test_path, category1, 0)
	#ave_percep.ave_percep(train_path, test_path, category1, 0)

	print '***Question 3: Implement stochastic gradient descent algorithm to minimize hinge loss***'
	#hinge_sgd.hinge_sgd(train_path, test_path, category1, 0)

	print '***Question 4: Multi-class Classification***'
	#multi_class.multi_classification(train_path, test_path, category2)

	print '***Question 5: Hinge loss with regularization term***'
	hinge_regularization.hinge_regularization(train_path, test_path, category1, 0)
	
	print '***Question 7: Logistic Regression***'
	#logistic_regression.logistic_reg(train_path, test_path, category1, 0)


if __name__ == "__main__":
	main()
	pass



