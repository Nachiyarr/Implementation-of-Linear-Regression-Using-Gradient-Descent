# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Import the required library and read the dataframe.
3. Write a function computeCost to generate the cost function.
4. Perform iterations og gradient steps with learning rate.
5. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:

#### Program to implement the linear regression using gradient descent.
#### Developed by: ALAGU NACHIYAR K
#### RegisterNumber:  212222240006
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)
```

### Output:
![Screenshot 2024-09-18 135216](https://github.com/user-attachments/assets/befb6dc7-7442-4e79-9c21-f114f9376c49)

```
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
```

### Output:
![Screenshot 2024-09-18 135223](https://github.com/user-attachments/assets/f4fddf32-11d5-4eba-a147-b9574c1d0bf3)

```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
```

### Output:
![Screenshot 2024-09-18 135228](https://github.com/user-attachments/assets/b1ab04ed-8de3-40be-94c1-22fd000dbd9d)

```
theta = linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")
```

### Output:
![Screenshot 2024-09-18 135232](https://github.com/user-attachments/assets/0713d2d2-5f3e-41ad-a406-64e772fca62f)




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
