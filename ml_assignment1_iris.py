from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris=load_iris()
import time
#output=iris.target_names[0]
#loading the output values for setosa,versicolor and virginica
setosa=iris.target[0:50]
versicolor=iris.target[50:100]
virginica=iris.target[100:150]

#training data(output values) for each of three flowers out of there are 49 records for training and one for testing
setosa_training=setosa[0:49]
versicolor_training=versicolor[0:49]
virginica_training=virginica[0:49]

#testing data(output values) for each flower each data contains only one record
setosa_testing=setosa[-1]
versicolor_testing=versicolor[-1]
virginica_testing=virginica[-1]

#this part contains the actual data
setosa_data=iris.data[0:50]
versicolor_data=iris.data[50:100]
virginica_data=iris.data[100:150]

#it contains the data to be used for training
setosa_data_training=setosa_data[0:49]
versicolor_data_training=versicolor_data[0:49]
virginica_data_training=virginica_data[0:49]

#it contains the data to be used for testing
setosa_data_testing=setosa_data[-1]
versicolor_data_testing=versicolor_data[-1]
virginica_data_testing=virginica_data[-1]

#creating input variable containg the data
input=np.concatenate((setosa_data_training,versicolor_data_training,virginica_data_training))
#print(input.shape)
#time.sleep(4)

#creating output variables containing the output(0,1,2)
output=np.concatenate((setosa_training,versicolor_training,virginica_training))
#print(output.shape)
#setosa_features=[setosa_data[0:49],setosa_training]
#versicolor_features=versicolor_data[0:49]
#virginica_features=virginica_data[0:49]

#fitting of datasets
setosa_algorithm=tree.DecisionTreeClassifier()
trained=setosa_algorithm.fit(input,output)

print(setosa_data_testing)
#predicted result
#result=trained.predict([[6.6 2.3 4.5 6.6]])
#print(result)

#setosa_trained=setosa_algorithm.fit(setosa_features,output)
#res=setosa_trained.predict([setosa_data[-1],setosa_testing])
#print(res)
#print(dir(iris))
#print(setosa_features)
#print(setosa_testing)
'''rint(setosa_features)
print(versicolor_features)
print(virginica_features)
print(output)'''
