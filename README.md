# Linear-Regression
Step 1: Import necessary libraries

import pandas as pd  - for working with dataset
import numpy as np   - numerical Python
import matplotlib.pyplot as plt - To visualise the results
from sklearn.model_selection import train_test_split - To split train and test data
from sklearn.linear_model import LinearRegression - Import the LinearRegression model

Step 2: Importing the dataset

X=dataset.iloc[:,:-1].values - Independent Variable X
y=dataset.iloc[:,-1].values  - Dependent Variable Y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=2)

Step 3: Fit and Predict

regressor=LinearRegression()
regressor.fit(X_train,y_train)
ypred=regressor.predict(X_train)

Step 4: Visualise the Results

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,ypred)
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

