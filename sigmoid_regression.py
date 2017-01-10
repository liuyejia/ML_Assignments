import assignment1 as as1
import numpy as np
#import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
#from sklearn.metrics import mean_squared_error
from math import sqrt
#import pandas as pd

countries,features,values=as1.load_unicef_data()

def sigmoid(x,u,s):
 	newX= 1 / (1 + np.exp((u-x)/s))
        return newX

x=np.reshape(np.array(values[:100,10]),100)
y=np.reshape(np.array(values[:100,1]),100)

newX1=sigmoid(x,100,2000)
newX2=sigmoid(x,10000,0)

#include bias term:matrix ones
x_y_combined=np.array(np.matrix([np.ones(newX1.size),newX1,newX2]))
#calcute coefficients
pinv=np.linalg.pinv(x_y_combined)
coefs=np.matrix(y)*pinv
intercept=coefs.item((0,0))
p1=coefs.item((0,1))
p2=coefs.item((0,2))

model_y=intercept+p1*newX1+p2*newX2

#sort sig.X to make plot a clear curve rather than several curves
newX1_sorted=np.sort(newX1)
newX2_sorted=np.sort(newX2)
model_y_curve=intercept+p1*newX1_sorted+p2*newX2_sorted;

#plot
plt.ylabel('y')
plt.xlabel('x')
plt.plot(np.sort(x),model_y_curve,'r.-')
plt.plot(np.sort(x),model_y_curve,'bo')
plt.title('sigmoid basis function regression')
plt.show()


#x2,y2 denote test data
x2=np.reshape(np.array(values[100:195,10]),95)
y2=np.reshape(np.array(values[100:195,1]),95)

#sigmoid test data
newX1_x2=sigmoid(x2,100,2000)
newX2_x2=sigmoid(x2,10000,0)

model_y2=intercept+p1*newX1_x2+p2*newX2_x2

#calculate train_error&test_error
#rms_train=sqrt(mean_squared_error(model_y, y))
rms_train=math.sqrt(np.power(model_y-y,2).sum()/train_res.size)
#rms_test=sqrt(mean_squared_error(model_y2,y2))
rms_test=math.sqrt(np.power(model_y2,y2,2).sum()/train_res.size)
print "train_error=",rms_train,"test_error=",rms_test



