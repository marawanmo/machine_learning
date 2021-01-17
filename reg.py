import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

# data voor het model
df = pd.read_csv("data.csv")

# print wat informatie over data
# print(df.head())
# print(df.describe())


# data van 4 tabellen
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# histogram van alle data
# cdf.hist()
# plt.show()


# maak scatter plot van emission en fuel. consump.
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()


# zorgt ervoor dat je 20 procent test data hebt en 80 procent training data
# np array met een distrobutie van 0 tot 1 met len len(df)
msk = np.random.rand(len(df)) < 0.8
# 80 & naar training data en 20% naar testing data
train = cdf[msk]
test = cdf[~msk]


# scatter plot van training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# zorgt voor linear regression model
regr = linear_model.LinearRegression()

# x is de data en y is de label die erbij hoort.
# as any array converts van pd naar numpy array
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

fitness = regr.fit(train_x, train_y)
print("coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)
new_x = int(input("What is your engine size? "))
print("Your emission might be: " + str(regr.coef_[0][0] * new_x + regr.intercept_[0]))


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
