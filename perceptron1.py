#import useful packages
import pandas #package for data analysis
pandas.set_option('max_rows', 10)
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
import statsmodels.api as statsmodels #useful stats package with linear regression functions
import seaborn as sns #very nice plotting package
sns.set(color_codes=True, font_scale=1.3)
sns.set_style("whitegrid")

#import data
filename = ('/users/ohsehun/downloads/soil_observations.csv')
data = pandas.read_csv(filename)
data
#turns the dataframe into a list for easier analysis
y2=np.array(list(data['OC5'].values))
x0=np.array(list(data['Clay1'].values)) 
x1=np.array(list(data['CEC2'].values))
            
x=np.zeros((147,3))
xx=np.zeros((147,3))

for i in range(147):
    x[i][0]=x0[i]
    x[i][1]=x1[i]
    x[i][2]=1
            
for i in range(147):
    xx[i][0]=1
    xx[i][1]=x0[i]
    xx[i][2]=x1[i]
    
#packages
import matplotlib.pyplot as plt
import numpy as np

def logic_perceptron(xx, y2):
    #loads the data
    filename='/users/ohsehun/downloads/soil_observations.csv'
    X_file = np.genfromtxt(filename, delimiter=',', skip_header=1)
    N = np.shape(X_file)[0]
    X = xx
    Y = y2

    # Standardize the input 
    X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])
    X[:, 2] = (X[:, 2]-np.mean(X[:, 2]))/np.std(X[:, 2])

    global w
    #weights of the bias weight and the features
    w = np.array([0, 0, 0])
    epoch = 100
    lc = 0.00001
    for t in range(0, epoch):
        #iterates over each data point for one epoch
        grad_t = np.array([0., 0., 0.])
        for i in range(0, N):
            x_i = X[i, :]
            y_i = Y[i]
            #dot product that computes the error
            h = np.dot(w, x_i)-y_i
            grad_t += x_i*h

        #updates the weights
        w = w - lc*grad_t
    print ("Weights found:",w)

def B(coefficients): #error function
    k = len(data) #assigns the length of the loaded array to a variable "k"
    tot = 0 #assigns 0 to the variable "tot"
    for j in range(k): #creates a loop that iterates k times
        y = coefficients[1] * x0[j] + coefficients[2]*x1[j]+coefficients[0] 
        #finds the predicted values of the response (y) by the formula: y=b1*x+b0, 
        #where b1 is the best solution for slope and b0 is the intercept
        res = y2[j] - y 
        #calculates the residuals by formula: res = observed value - predicted value 
        tot += res**2 #finds the sum of the squared residuals, which is the error of the regression line
    return tot/k #returns the error 
    
logic_perceptron(xx, y2)
print("The error is", B(w)) #prints out the error
