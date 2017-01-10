import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv
import assignment1 as as1
 
countries,features,values=as1.load_unicef_data() 
validation_error=[]

#dataset=values[:100,7:40]
dataset_norm=as1.normalize_data(values[:,7:40])
t=values[:100,1]
p=np.array([0.01,0.1,1,10,100,1000,10000])

 
def monomial_polyFunction(x,w):
    return np.dot(x,w)


#dataset_L2=np.concatenate((np.array(dataset),np.array(np.power(dataset,2))),axis=1)
#I_matrix=np.identity(66)

for p_item in p:
    t_cross_error=[]
    for i in range(1,11):
        
        validation_data =dataset_norm[(i-1)*10:i*10,:]
        validation_t=values[(i-1)*10:i*10,1]
        if i==1:
            train_data =dataset_norm[10:100, :]
            train_t=values[10:100,1]
        elif i== 10:
            train_data=dataset_norm[:90,:]
            train_t=values[:90,1]
        else:
            train_data=np.concatenate((dataset_norm[:(i-1)*10,:],dataset_norm[i*10:100,:]),axis=0)
            train_t=np.concatenate((values[:(i-1)*10,1],values[i*10:100,1]),axis=0)
       #print train_data.shape
       #print train_t.shape
       #p_item*I_matrix+(train_data.transpose())*train_data))*(train_data.transpose())*(np.array(t))
        
        train_data_L2=np.concatenate((np.array(train_data),np.power(np.array(train_data),2)),axis=1)
        
        train_data_L2=np.concatenate((np.transpose(np.matrix(np.ones(90))),train_data_L2),axis=1)
        
        validation_data_L2=np.concatenate((np.array(validation_data),np.power(np.array(validation_data),2)),axis=1)
        validation_data_L2=np.concatenate((np.transpose(np.matrix(np.ones(10))),validation_data_L2),axis=1)
   
        inverse= inv(np.transpose(train_data_L2).dot(train_data_L2)+ p_item * np.identity(np.transpose(train_data_L2).dot(train_data_L2).shape[1]))
        w=inverse.dot(np.transpose(train_data_L2)).dot(train_t)
       
        t_res=monomial_polyFunction(validation_data_L2,w)
        rms_validation=math.sqrt(np.power(validation_t-t_res,2).sum()/validation_t.size)
        #rms_validation=math.sqrt((np.power(validation_t-t_res,2).sum()+p_item*np.transpose(w).dot(w))/validation_t.size)

        #print rms_validation
        t_cross_error.append(rms_validation)
            
    error_p=(float(sum(t_cross_error))/len(t_cross_error))
    validation_error.append(error_p)



#print validation_error
plt.semilogx(p,validation_error)
plt.ylabel('validation_error')
plt.title('cross_validation')
plt.xlabel('lambda')
plt.show()

