# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 12:05:11 2019

@author: Albert
"""

#import pystan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from Mymodel1 import Performancematrix
from Mymodel1 import plotk,normcdf,uni,plotm
from sklearn import preprocessing
from uno2p import uno2p2,uno2p1,uno2p3
import copy
import pandas as pd

def makeprediction(pre):
    l=np.zeros(pre.shape[0])
    for i in range(pre.shape[0]):
        
        if(pre[i]>0.5):
            l[i]=1
        else:
            l[i]=0
    return l;
 
x1=np.arange(0)
x2=np.empty(0)
for i in range(400):
    x1=np.append(x1,np.arange(400))
    x2=np.append(x2,np.repeat(i,400))
    
X=np.vstack((x1,x2)).T
X2=copy.deepcopy(X)
y=np.zeros(160000)
for i in range(160000):
    k1=X[i,0]//100
    k2=X[i,1]//100
    d=np.abs(k1-k2)
    if((d%2)==0):
        y[i]=1
    else:
        y[i]=0
      
y=np.expand_dims(y,axis=1)        
m=y.reshape(400,400)


plt.matshow(m, cmap='ocean')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()
#T=np.c_[X,y]
X=preprocessing.scale(X)
y=np.squeeze(y)
#sm=MCMC()
accuracy1=np.empty(0)
accuracy2=np.empty(0)
accuracy3=np.empty(0)
X1=X2[:,0]
Xtrain1=X2
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]  
ytrain=y
diff=np.ones(y.shape[0])/np.sum(np.ones(y.shape[0]))
table_500_300={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_300)
table1.to_csv("table_full.csv",header=True)
"""

l=np.random.randint(0,y.shape[0]-1,200)
Xtrain=X[l,:]
ytrain=y[l]
Xtrain1=X2[l,:]

yshow=np.zeros(160000)
for i in range(Xtrain1.shape[0]):
    t=Xtrain1[i,0]*400+Xtrain1[i,1]
    t=t.astype(int)
    yshow[t]=1
mtrain=yshow.reshape(400,400)
plt.matshow(mtrain,cmap='winter')
plt.colorbar()
plt.show()
"""
l1=np.random.randint(0,y.shape[0]-1,16000)
Xtest=X[l1,:]
ytest=y[l1]
forest=RandomForestClassifier(n_estimators=500)
forest.fit(Xtrain,ytrain)
a1=forest.score(Xtest,ytest) 
accuracy1=np.append(accuracy1,a1)
ntree=1000
C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
C=C.astype(int)
C=C.T
#C=C.T
#P=(np.repeat(30,C.shape[0])-np.sum(C,axis=1))*3
#data1={"np":ntree,"ni":Xtrain.shape[0],"U":C,"prior":P}
#data1={"np":ntree,"ni":Xtrain.shape[0],"U":C}
#fit = sm.sampling(data=data1, iter=300, chains=1,n_jobs=1)

##500 classifier 50 iter
ae,be,the=uno2p3(C,20)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_500_50={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_50)
table1.to_csv("table_500_50.csv",header=True)
##500 classifier 300 iter
ae,be,the=uno2p3(C,300)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_500_300={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_300)
table1.to_csv("table_500_300.csv",header=True)
##500 clasifier 1000 iter
ae,be,the=uno2p3(C,1000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_500_1000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_1000)
table1.to_csv("table_500_1000.csv",header=True)

##500 classifier 2000 iter
ae,be,the=uno2p3(C,2000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_500_2000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_2000)
table1.to_csv("table_500_2000.csv",header=True)


##pic2



def makeprediction(pre):
    l=np.zeros(pre.shape[0])
    for i in range(pre.shape[0]):
        
        if(pre[i]>0.5):
            l[i]=1
        else:
            l[i]=0
    return l;
x1=np.arange(0)
x2=np.empty(0)
for i in range(400):
    x1=np.append(x1,np.arange(400))
    x2=np.append(x2,np.repeat(i,400))   
X=np.vstack((x1,x2)).T
X2=copy.deepcopy(X)
y=np.zeros(160000)
for i in range(160000):
    k1=X[i,0]//100
    k2=X[i,1]//100
    d=np.abs(k1-k2)
    if((d%2)==0):
        y[i]=1
    else:
        y[i]=0     
y=np.expand_dims(y,axis=1)        
m=y.reshape(400,400)
plt.matshow(m, cmap='ocean')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()
X=preprocessing.scale(X)
y=np.squeeze(y)
accuracy1=np.empty(0)
accuracy2=np.empty(0)
accuracy3=np.empty(0)
l=np.random.randint(0,y.shape[0]-1,200)
Xtrain=X[l,:]
ytrain=y[l]
Xtrain1=X2[l,:]


###100 classifier 2000 iter
l1=np.random.randint(0,y.shape[0]-1,16000)
Xtest=X[l1,:]
ytest=y[l1]
forest=RandomForestClassifier(n_estimators=30)
forest.fit(Xtrain,ytrain)
a1=forest.score(Xtest,ytest) 
accuracy1=np.append(accuracy1,a1)
ntree=30
C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
C=C.astype(int)
C=C.T
ae,be,the=uno2p3(C,1000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_100_2000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_100_2000)
table1.to_csv("table_100_2000.csv",header=True)



###300 classifier 2000 iter
l1=np.random.randint(0,y.shape[0]-1,16000)
Xtest=X[l1,:]
ytest=y[l1]
forest=RandomForestClassifier(n_estimators=100)
forest.fit(Xtrain,ytrain)
a1=forest.score(Xtest,ytest) 
accuracy1=np.append(accuracy1,a1)
ntree=100
C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
C=C.astype(int)
C=C.T
ae,be,the=uno2p3(C,1000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_300_2000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_300_2000)
table1.to_csv("table_300_2000.csv",header=True)



###500 classifier 2000 iter
l1=np.random.randint(0,y.shape[0]-1,16000)
Xtest=X[l1,:]
ytest=y[l1]
forest=RandomForestClassifier(n_estimators=500)
forest.fit(Xtrain,ytrain)
a1=forest.score(Xtest,ytest) 
accuracy1=np.append(accuracy1,a1)
ntree=500
C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
C=C.astype(int)
C=C.T
ae,be,the=uno2p3(C,1000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_500_2000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_500_2000)
table1.to_csv("table_500_2000.csv",header=True)

###1000 classifier 2000 iter
l1=np.random.randint(0,y.shape[0]-1,16000)
Xtest=X[l1,:]
ytest=y[l1]
forest=RandomForestClassifier(n_estimators=1000)
forest.fit(Xtrain,ytrain)
a1=forest.score(Xtest,ytest) 
accuracy1=np.append(accuracy1,a1)
ntree=1000
C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
C=C.astype(int)
C=C.T
ae,be,the=uno2p3(C,2000)
diff=np.average(be,axis=0)
cdfdiff=normcdf(diff)
diff=cdfdiff/np.sum(cdfdiff)
X1=Xtrain1[:,0]
X2=Xtrain1[:,1]    
table_1000_2000={"X1":X1,"X2":X2,"Y":ytrain,"diff":diff}
table1=pd.DataFrame(table_1000_2000)
table1.to_csv("table_1000_2000.csv",header=True)


"""
import matplotlib.cm as cm

fig, ax = plt.subplots()
ax.scatter(X1, X2, c=ytrain*1500, s=diff*6000, alpha=0.9,cmap='winter')
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)
ax.set_title('Illustration of the difficulties 3000 iter with 1000 classifier')
ax.grid(False)
plt.xticks([])
plt.yticks([])
ax.xaxis.set_major_locator(plt.MultipleLocator(100))#设置x主坐标间隔 1
ax.yaxis.set_major_locator(plt.MultipleLocator(100))#设置y主坐标间隔 1
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')

plt.show()
"""
###whole graph
X11=X2[:,0]
X12=X2[:,1]
fig, ax = plt.subplots()
ax.scatter(X11, X12, c=y*100, alpha=0.9,cmap='winter')
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)
ax.set_title('Chess Board')
ax.grid(False)
ax.xaxis.set_major_locator(plt.MultipleLocator(100))#设置x主坐标间隔 1
ax.yaxis.set_major_locator(plt.MultipleLocator(100))#设置y主坐标间隔 1
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
plt.show()







disc=np.average(ae,axis=0)
cdfdiff=normcdf(disc)
disc=disc/np.sum(disc)
fig, ax = plt.subplots()
ax.scatter(X1, X2, c=ytrain*100, s=disc*6000, alpha=0.9,cmap='winter')
ax.set_xlabel("X", fontsize=15)
ax.set_ylabel("Y", fontsize=15)
ax.set_title('Illustration of the difficulties')
ax.grid(False)
ax.xaxis.set_major_locator(plt.MultipleLocator(100))#设置x主坐标间隔 1
ax.yaxis.set_major_locator(plt.MultipleLocator(100))#设置y主坐标间隔 1
ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
plt.show()



l=np.average(the,axis=0)
l1=normcdf(l)
w1=l1/np.sum(l1)
pre2=np.dot(T,w1.T)
pre2=makeprediction(pre2)
a2=1-np.sum(np.abs(pre2-np.squeeze(ytest)))/Xtest.shape[0]
accuracy2=np.append(accuracy2,a2)
"""
l2=uni(l)
w2=l2/np.sum(l2)
pre3=np.dot(T,w2.T)
pre3=makeprediction(pre3)
a3=1-np.sum(np.abs(pre3-np.squeeze(ytest)))/Xtest.shape[0]
accuracy3=np.append(accuracy3,a2)
"""
print("finish",i)