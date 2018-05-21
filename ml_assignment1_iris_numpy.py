from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import time

#loading iris dataset
iris=load_iris()

#list containing end points
x=[0,50,100]

#removing of data at position 0,50,100
train_data=np.delete(iris.data,x,axis=0)

#removing of target at position 0,50,100
train_target=np.delete(iris.target,x,axis=0)

#generating test data for position 0,50,100
test_data=iris.data[x]

#generating test target for position 0,50,100
test_target=iris.target[x]

#Defining the decision tree classifier
clf=tree.DecisionTreeClassifier()

#training of classifier
trained=clf.fit(train_data,train_target)

#testing with the test data and test target
predicted=trained.predict(test_data)
print(predicted)
