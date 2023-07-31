# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 18:15:46 2023

@author: upyou
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import chdir
from scipy.optimize import minimize

chdir("C:/Users/upyou/OneDrive/Desktop/coding practice.1")
dataRUN = pd.read_csv("mileRecords.csv")

x = np.array(dataRUN["year"])
x = x-x[0]
y = np.array(dataRUN["time"])

plt.plot(x,y, 'xk')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.title('Mile Record Time')

def fmin(param):
    K=param[0]
    r=param[1]
    e=y-r*y*(1-y/K)*x
    return(np.dot(e,e.T))
def f(y,K,r,x):
    return(r*y*(1-y/K)*x)


x0=(3.1,-.25)
opt=minimize(fmin,x0)
print(opt)
print(opt.x[0])
xaxis=np.linspace(0,90,num=32)

plt.plot(x,y, 'xk')
plt.xlabel('Year')
plt.ylabel('Time in seconds')
plt.title('Mile Record Time')
K=3.6
r=-.27
timepredict=f(xaxis, K, r, y)
plt.plot(xaxis, timepredict, color='red')