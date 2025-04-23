# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: TRISHA PRIYADARSHNI PARIDA
RegisterNumber:  212224230293

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()

*/
```

## Output:


![Screenshot 2025-04-23 112803](https://github.com/user-attachments/assets/48bcb20f-1188-4865-b4d6-e4c876432773)


![Screenshot 2025-04-23 112809](https://github.com/user-attachments/assets/b12369ee-fdc3-47f9-ba66-68801f1b1698)


![Screenshot 2025-04-23 112813](https://github.com/user-attachments/assets/330ecbca-ba74-48f2-b0e8-f96fea30a29b)


![Screenshot 2025-04-23 112818](https://github.com/user-attachments/assets/a75815a2-1139-4634-99e5-1693489a3683)


![Screenshot 2025-04-23 112825](https://github.com/user-attachments/assets/38762ef5-136d-44ba-b56c-32df6ae6f463)


![Screenshot 2025-04-23 112903](https://github.com/user-attachments/assets/a0c35f40-fcd4-4f2f-b2b0-6e73accf6899)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
