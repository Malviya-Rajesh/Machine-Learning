# Importing useful Library
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
x = np.random.randn(10,2)
x[:,1:] = x[:,1:]/x[:,1:]
y = (x[:,:1] * 0.5) + len(x)

x_train = x[:8,:]
x_test = x[8:,:]
y_train = y[:8,:]
y_test = y[8:,:]


# model define for simple linear regretion
def model(x, y, learing_rate, iterations):
    m = len(y)
    theta = np.zeros((2,1))
    cost_list = []

    for i in range(iterations):
        y_pred = np.dot(x,theta)
        cost = (1/(2*m)) * (np.sum(np.square(y_pred-y)))
        d_theta =  (1/m) * (np.dot(x.T,(y_pred - y)))
        theta = theta - (learing_rate * d_theta)
        cost_list.append(cost)

    return cost_list, theta

cost_list, theta = model(x_train, y_train, learing_rate = 0.3, iterations = 50) 

#print(cost_list)
#print(theta)

predication = np.dot(x_test, theta)

#print(predication)
#print(y_test)


#mrse = 0.5 * np.sqrt((np.sum(np.square(predication - y_test)))) # Where mrse is "Mean Root Squre Error" (Here we know y_test have 2 elemnts so 1/m = 0.5)
#mse = 0.5 * (np.sum(np.square(predication - y_test))) # Where mse is "Mean Squre Error" (Here we know y_test have 2 elemnts so 1/m = 0.5) 
#print(mrse,"\n",mse)
# output :=> very very  low error

def score(predication,y):
    in_function_socre = np.sum(np.abs(((predication/y)*100)-100)) / len(y)
    return 100 - in_function_socre

scores = score(predication,y_test)

print(scores)

plt.scatter(predication,y_test)
plt.show()
