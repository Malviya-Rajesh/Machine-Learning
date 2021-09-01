import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 

X = np.ones((10,2))
X[:,:1] = X[:,:1] * np.random.randint(0,10,(10,1))

y = np.ones((10,1))

for i in range(0,10):
    if X[i][0] >= 5:
        y[i] = 0

print(X,"\n",y)

x_train = X[:7,:]
x_test = X[7:,:]
y_train = y[:7,:]
y_test = y[7:,:]

def model(x, y, alpha, loop):
    m = len(y)
    theta = np.zeros((2,1))
    cost_lst = []

    for i in range(loop):
        function_of_z = np.dot(theta.T,x.T)
        y_pred = 1 / (1 + np.exp(-function_of_z))
        cost = (-1/m) * (np.sum(np.dot(y,np.log(y_pred)) + np.dot((1-y), np.log(1-y_pred))))
        d_theta = (1/m) * (np.sum(np.dot(y_pred-y,x)))
        theta = theta - (alpha * d_theta)
        cost_lst.append(cost)

    return theta, cost_lst

theta, cost_lst = model(x_train,y_train,0.3,50)

print(theta)

function_of_z = np.dot(theta.T,x_test.T)
y_pred = 1 / (1 + np.exp(-function_of_z))


for i in range(0,len(y_test)):
    if y_pred[0][i] >= 0.5:
        y_pred[0][i] = 1
    else:
        y_pred[0][i] = 0

print(y_pred)



plt.scatter(y_pred,y_test)
plt.show()