import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

c_data = datasets.load_breast_cancer()
print(c_data.feature_names)
print(c_data.target_names)

X = c_data.data
y = c_data.target 

#split data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
classes = ['malignant' 'benign']

model = svm.SVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)

print(acc)