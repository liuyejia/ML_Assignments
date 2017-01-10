import assignment1 as as1
import numpy as np

import math
import matplotlib.pyplot as plt
 
train_error=[]
test_error=[]
countries,features,values=as1.load_unicef_data()
values_norm=as1.normalize_data(values[:,7:40])

train_data=values_norm[:100,:]
test_data=values_norm[100:195,:]
train_y=values[:100,1]
test_y=values[100:195,1]
#train_data=as1.normalize_data(train_data)
#test_data=as1.normalize_data(test_data)

def monomial_polyFunction(w,x):
   return x*w

for degree in range(1,7):
   train_data_degree=train_data
   test_data_degree=test_data
   if(degree!=1):
        for i in range(2,degree+1):
           train_data_degree=np.concatenate((np.array(train_data_degree),np.array(np.power(train_data,i))),axis=1)
           test_data_degree=np.concatenate((np.array(test_data_degree),np.array(np.power(test_data,i))),axis=1)
   train_data_degree=np.concatenate((np.transpose(np.matrix(np.ones(100))),train_data_degree),axis=1)
   test_data_degree=np.concatenate((np.transpose(np.matrix(np.ones(95))),test_data_degree),axis=1)

   pinv=np.linalg.pinv(train_data_degree)
   coefs=pinv*np.matrix(train_y)
   train_res=monomial_polyFunction(coefs,train_data_degree)
   test_res=monomial_polyFunction(coefs,test_data_degree)
   
   #rms_train=sqrt(mean_squared_error(train_res,train_y))
   #rms_test=sqrt(mean_squared_error(test_res,test_y))
   rms_train=math.sqrt((np.power(train_res-train_y,2).sum())/train_res.size)
   rms_test=math.sqrt((np.power(test_res-test_y,2).sum())/test_res.size)
   train_error.append(rms_train)
   test_error.append(rms_test)

degree=np.arange(1,7)
plt.plot(degree,train_error)
plt.plot(degree,test_error)
plt.ylabel('RMS')
plt.legend(['Train error','Test error'])
plt.title('Fit with polynomials with regularization')
plt.xlabel('Polynomial degree')
plt.show()




 
