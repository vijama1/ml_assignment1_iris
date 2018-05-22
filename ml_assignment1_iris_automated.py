import time
from sklearn.datasets import load_iris
from sklearn import tree

from sklearn.metrics import accuracy_score

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
#print(help(train_test_split))

iris=load_iris()

train_data,test_data,train_target,test_target=train_test_split(iris.data,iris.target,test_size=0.5)

clf=tree.DecisionTreeClassifier()
trained=clf.fit(train_data,train_target)

predicted=trained.predict(test_data)
print(predicted)

accuracy=accuracy_score(test_target,predicted)
print(accuracy)
