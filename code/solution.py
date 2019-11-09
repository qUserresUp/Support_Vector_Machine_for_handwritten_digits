import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict

'''s
Homework4: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''

def svm_with_diff_c(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different value 
	of cost c to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The value of cost c you need to try is listing as follow:
	c = [0.01, 0.1, 1, 2, 3, 5]
	Please keep other parameter options as default.
	No return value is needed
	'''
	
	m = svm_train(train_label, train_data, '-c 3')
	"""
	m = svm_train(train_label, train_data, '-c 0.01')
	m = svm_train(train_label, train_data, '-c 0.1')
	m = svm_train(train_label, train_data, '-c 1')
	m = svm_train(train_label, train_data, '-c 2')
	m = svm_train(train_label, train_data, '-c 5')

	"""
	p_label, p_acc, p_val = svm_predict(test_label, test_data, m)



	"""
	print("p_label: \n", p_label)
	print("p_acc: \n", p_acc)
	print("p_val: \n", p_val)
	"""


def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different kernel
	to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The kernel  you need to try is listing as follow:
	1. linear kernel
	2. polynomial kernel
	3. radial basis function kernel
	Please keep other parameter options as default.
	No return value is needed
	'''


	m = svm_train(train_label, train_data, '-t 2')
	"""
	m = svm_train(train_label, train_data, '-t 0')
	m = svm_train(train_label, train_data, '-t 1')	

	"""
	p_label, p_acc, p_val = svm_predict(test_label, test_data, m)



