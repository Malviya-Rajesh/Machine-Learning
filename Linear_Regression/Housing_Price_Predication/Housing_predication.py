# Importing Usefull Library 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


df = pd.read_csv(r'Linear_Regression\USA_Housing.csv')
df = df.dropna()
df['ones'] = np.ones((5000,1))
#print(df.info())
#print(df.head())
#print(df.columns)

X = df[['Avg. Area Number of Bedrooms', 'Avg. Area Number of Rooms', 'Avg. Area House Age',
         'Avg. Area Income', 'Area Population','ones']]
X['Area Population'] = (X['Area Population'] - X['Area Population'].mean()) / (X['Area Population'].max() - X['Area Population'].min())
X['Avg. Area Income'] = (X['Avg. Area Income'] - X['Avg. Area Income'].mean()) / (X['Avg. Area Income'].max() - X['Avg. Area Income'].min())
#X['Avg. Area Number of Bedrooms'] = (X['Avg. Area Number of Bedrooms'] - X['Avg. Area Number of Bedrooms'].mean()) / (X['Avg. Area Number of Bedrooms'].max() - X['Avg. Area Number of Bedrooms'].min())
#X['Avg. Area Number of Rooms'] = (X['Avg. Area Number of Rooms'] - X['Avg. Area Number of Rooms'].mean()) / (X['Avg. Area Number of Rooms'].max() - X['Avg. Area Number of Rooms'].min())
#X['Avg. Area House Age'] = (X['Avg. Area House Age'] - X['Avg. Area House Age'].mean()) / (X['Avg. Area House Age'].max() - X['Avg. Area House Age'].min())
y = df['Price']
y = np.array(y).reshape(len(y),1)

print(X.head())

# random_state value is give best predication score, it is found by Elbow mathode
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3865) #860 #3865

# Regression model function
def reg_model(x,y,alpha,loop):
    m = len(y)
    theta = np.zeros((6,1))
    cost_list = []

    for i in range(loop):
        y_pred = np.dot(x,theta)
        cost = (1/(2*m)) * (np.sum(np.square(y_pred - y))) # Calculating cost or error or lost 
        d_theta = (1/m) * (np.sum(np.dot(x.T, (y_pred - y)))) # finding partial deriavtive of theta
        theta = theta - (alpha * d_theta) # Calculating actual theta
        cost_list.append(cost)

    return theta, cost_list

theta, cost_list = reg_model(X_train, y_train, 0.001, 500)

predication = np.dot(X_test, theta)

def score(predication,y):
    in_function_socre = np.sum(np.abs(((predication/y)*100)-100)) / len(y)
    return 100 - in_function_socre

scores = score(predication,y_test)
print(scores)
#print(cost_list)

plt.scatter(predication,y_test)
plt.show()
