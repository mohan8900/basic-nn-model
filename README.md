#EX NO:1
# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

## Neural Network Model

![image](https://user-images.githubusercontent.com/75234942/187088677-90eab090-03e2-46f9-ad99-3240f9753719.png)




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data1 = pd.read_csv('exp1.csv')
data1.head()
X = data1[['input']].values
X
Y = data1[["output"]].values
Y
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scalar=MinMaxScaler()
scalar.fit(X_train)
scalar.fit(X_test)
X_train=scalar.transform(X_train)
X_test=scalar.transform(X_test)
import tensorflow as tf
model=tf.keras.Sequential([tf.keras.layers.Dense(4,activation='relu'),
                          tf.keras.layers.Dense(5,activation='relu'),
                          tf.keras.layers.Dense(1)])
model.compile(loss="mae",optimizer="rmsprop",metrics=["mse"])
history=model.fit(X_train,Y_train,epochs=1000)
import numpy as np
X_test
preds=model.predict(X_test)
np.round(preds)
tf.round(model.predict([[20]]))
pd.DataFrame(history.history).plot()
r=tf.keras.metrics.RootMeanSquaredError()
r(Y_test,preds)
```
## Dataset Information

![image](https://user-images.githubusercontent.com/75234942/187087432-f583dd0b-74f1-4ea3-98bb-ba54a7b3fc7b.png)




## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/75234942/187087268-5cd838d4-2f89-4fef-b590-35e1d879921f.png)


### Test Data Root Mean Squared Error
![image](https://user-images.githubusercontent.com/75234942/187087300-9a9eafea-bd75-4de3-b87a-6ccd9b6241c5.png)



### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/75234942/187087345-45cb2305-c766-4ec6-8db6-3e74a6993643.png)


## RESULT
Thus to develop a neural network model for the given dataset has been implemented successfully.
