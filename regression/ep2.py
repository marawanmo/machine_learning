import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# onze gehele data
data = pd.read_csv(r"C:\Users\Katko\Desktop\NNFS\ML\data\student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# de waarden die we willen predicten
predict = "G3"

# X = features, y = labels
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# split je data in test en train samples
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

# onze model
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)

# hoe goed onze model gescored heeft
acc = linear.score(x_test, y_test)

print("CO:" , linear.coef_)
print("Inter:", linear.intercept_)

predict = linear.predict(x_test)
for x in range(len(predict)):
    print(predict[x], x_test[x], y_test[x])