# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start

Step 2: Import the required library and read the dataframe.

Step 3: Write a function computeCost to generate the cost function.

Step 4: Perform iterations og gradient steps with learning rate.

Step 5: Plot the Cost function using Gradient Descent and generate the required graph.

Step 6: Stop


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Maheswaran K
RegisterNumber:  212222110023
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:/Users/admin/Downloads/VARSHINI/50_Startups.csv")
data.head()
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")
```

## Output:


![ml2](https://github.com/anu-varshini11/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138969827/475640c7-9efb-4ca2-882a-aa52906026bd)
![ml3](https://github.com/anu-varshini11/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138969827/c81d7ffe-da92-491d-912d-9768469b2f81)
![Screenshot 2024-04-27 114238](https://github.com/VARSHINI22009118/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119401150/30676fc4-7c69-4dd7-b9c1-a7ecbf69769b)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
