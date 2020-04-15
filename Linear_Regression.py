import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)
regressor=LinearRegression()
regressor.fit(X_train,y_train)
ypred=regressor.predict(X_train)
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,ypred)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
