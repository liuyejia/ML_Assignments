import assignment1 as as1
import numpy as np
#from sklearn.metrics import mean_squared_error
import math 
import matplotlib.pyplot as plt
 
train_error=[]
test_error=[]
countries,features,values=as1.load_unicef_data()

def monomial_polyFunction(w,x):
   return x*w

for i in range(7,15):
   #reshape input,target vector to 1D array
   x=np.array(values[:100,i])
   testData=values[100:195,i]
   test_y=np.array(values[100:195,1])
   train_y=np.array(values[:100,1])
   #polynomial regression
   #coefs=np.polyfit(x,y,3)
   #predict_onTrain=np.polyval(coefs,values[:100,i])
   train_data_degree=x
   test_data_degree=testData
   
   for i in range(0,4):
           train_data_degree=np.concatenate((np.array(train_data_degree),np.array(np.power(x,i))),axis=1)
           test_data_degree=np.concatenate((np.array(test_data_degree),np.array(np.power(testData,i))),axis=1)
   #pinv=np.linalg.pinv(np.array((np.matrix([np.ones(train_data_degree.size),train_data_degree]))))
   # coefs=np.matrix(y)*pinv
   #train_data_degree=np.array(np.matrix([np.ones(x.size),train_data_degree]))
   #calcute coefficients
   pinv=np.linalg.pinv(train_data_degree)
   coefs=pinv*np.matrix(train_y)
   train_res=monomial_polyFunction(coefs,train_data_degree)
   test_res=monomial_polyFunction(coefs,test_data_degree)
   
   #rms_train = sqrt(mean_squared_error(trainData,predict_onTrain))
   rms_train=math.sqrt(np.power(train_y-train_res,2).sum()/train_res.size)
   #predict_onTest=np.polyval(coefs,values[100:195,i])
   
   #rms_teist=sqrt(mean_squared_error(testData, predict_onTest))
   rms_test=math.sqrt(np.power(test_res-test_y,2).sum()/test_res.size)
   train_error.append(rms_train)
   test_error.append(rms_test)
 
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
 
     
#plot train_error&test_error by bar chart
#train_error plot
train_errort=tuple(train_error)
test_errort=tuple(test_error)
ind = np.arange(len(train_error))  # the x locations for the groups
width = 0.35       # the width of the bars
fig, ax = plt.subplots()
rects1= ax.bar(ind, train_errort, width, color='y')
rects2 = ax.bar(ind + width, test_error, width, color='r') 
# add some text for labels, title and axes ticks
ax.set_ylabel('errors')
ax.set_title('train_errors&test_errors')
ax.set_xticks(ind+width)
ax.set_xticklabels(('Feat_8', 'Feat_9', 'Feat_10', 'Feat_11', 'Feat_12','Feat_13','Feat_14','Feat_15'))
 

#test_error plot


autolabel(rects1)
autolabel(rects2)
ax.legend(['train_error','test_error'])
plt.show()
