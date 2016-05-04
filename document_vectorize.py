# Author: Zou Yanyan
#
# Function: load and vectorize files
#
#


from numpy import *
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import *

#category = ['atheism', 'sports']
#category2 = ['atheism', 'sports','science','politics']


#path1 = '/Users/Zoe/Desktop/HW1/data/train'

#path2 = '/Users/Zoe/Desktop/HW1/data/test'

#train_set = datasets.load_files(path1,categories=category1, 
#	load_content=True, shuffle=True, encoding='utf-8', decode_error='ignore', random_state=0)

#test_set = datasets.load_files(path2,categories=None, 
#	load_content=True, shuffle=True, encoding='utf-8', decode_error='ignore', random_state=0)

def createDataSet(train_path,test_path,category,k):
	"""
	create vectorized text feature
    '0' refer to 'atheism'
    '1' refer to 'sports'

	"""
	train_set = datasets.load_files(train_path,categories=category, 
	load_content=True, shuffle=True, encoding='utf-8', decode_error='ignore', random_state=0)

	count_vect = CountVectorizer(encoding = 'utf-8',lowercase = True,
	 decode_error = 'ignore',  analyzer = 'word', ngram_range = (2,4),min_df = 1)
	
	tfidf_vecter = TfidfVectorizer( max_df = 0.8, stop_words = 'english')

	test_set = datasets.load_files(test_path,categories=category, 
	load_content=True, shuffle=True, encoding='utf-8',  decode_error='ignore', random_state=0)

	

	X_train_tfidf = tfidf_vecter.fit_transform(train_set.data)
	X_train_counts = count_vect.fit_transform(train_set.data)

	X_test_tfidf = tfidf_vecter.transform(test_set.data)
	X_test_counts = count_vect.transform(test_set.data)


	 
	for i in range(X_train_counts.shape[0]):
		if train_set.target[i] == k:
			train_set.target[i] = 1
		else:
			train_set.target[i] = -1

	for i in range(X_test_counts.shape[0]):
		if test_set.target[i] == k:
			test_set.target[i] = 1
		else:
			test_set.target[i] = -1

	
	
	#X_train_normalize = preprocessing.normalize(X_train_counts, norm = 'l2')
	



	#print train_set.target_names
	#print train_set.target
	#print size 
	#print len(train_set.target)


	#print X_train_tfidf.shape
	#print X_train_counts
	#print X_train_normalize


	return X_train_counts, train_set.target, X_train_counts.shape,X_test_counts, test_set.target, X_test_counts.shape






#print train_set.target_names[0]  # '0' stands for 'atheism','1' stands for 'sports'
#print train_set.target_names[1]
#print train_set.target_names[2]
#print train_set.target_names[3]  # '0' stands for 'atheism','1' stands for 'politics', '2' science, '3' sports

#print len(train_set.target)
#print X_train_counts.shape

#print len(train_set.data)
#print len(train_set.filenames)
#print("\n".join(train_set.data[0].split("\n")[:3]))
#print(train_set.target_names[train_set.target[0]])
#print X_train_counts
#[x,y,z]=createDataSet(path1,category1)
#print x,y,z

#w = zeros(z[1])
#b =0



#createDataSet(path1,path2,category,0)






















