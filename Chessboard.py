# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 23:09:48 2018

@author: Albert
"""
import pystan
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
import random
#from uno3p import uno

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
"""
plt.matshow(m, cmap='ocean')
plt.colorbar()
plt.show()
"""
#T=np.c_[X,y]
X=preprocessing.scale(X)
y=np.squeeze(y)
#sm=MCMC()
accuracy1=np.empty(0)
accuracy2=np.empty(0)
accuracy3=np.empty(0)

####Baggint

seed=[2006,2007,8,19,27,36,58,57,99,71,72,73,91]

for i in seed:
    random.seed(i)
    l=np.random.randint(0,y.shape[0]-1,200)
    Xtrain=X[l,:]
    ytrain=y[l]
    random.seed(i)
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
    #C=C.T
    #P=(np.repeat(30,C.shape[0])-np.sum(C,axis=1))*3
    #data1={"np":ntree,"ni":Xtrain.shape[0],"U":C,"prior":P}
    #data1={"np":ntree,"ni":Xtrain.shape[0],"U":C}
    #fit = sm.sampling(data=data1, iter=300, chains=1,n_jobs=1)
    ae,be,the=uno2p3(C,1000)
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
    

np.savetxt("Checkseed.csv",seed)
np.savetxt("checkRF.csv",accuracy1)
np.savetxt("checkmy.csv",accuracy2)

#plotk(accuracy1.shape[0],accuracy1,accuracy2)
#plotm(accuracy1.shape[0],accuracy1,accuracy2,accuracy3)

gbr1=np.empty(0)
sv1=np.empty(0)
ds1=np.empty(0)
clfb1=np.empty(0)
lda1=np.empty(0)   
for i in range(15):
    ntree=500
    Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
    gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
    sv = svm.LinearSVC(random_state=i).fit(Xtrain,ytrain)
    ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
    clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
    lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
    acc_gbr=gbr.score(Xtest,ytest)
    gbr1=np.append(gbr1,acc_gbr)
    print("gbdt",acc_gbr)
    acc_sv=sv.score(Xtest,ytest)
    sv1=np.append(sv1,acc_sv)
    print("svm",acc_sv)
    acc_ds=ds.score(Xtest,ytest)
    ds1=np.append(ds1,acc_ds)
    print("ds",acc_ds)
    acc_clfb=clfb.score(Xtest,ytest)
    clfb1=np.append(clfb1,acc_clfb)
    print("clfb",acc_clfb)
    acc_lda=lda.score(Xtest,ytest)
    lda1=np.append(lda1,acc_lda)
    print("lda",acc_lda)
 
print(np.mean(gbr1))

    
df=pd.DataFrame()
df["GBDT"]=gbr1
df["SVM"]=sv1
df["Tree"]=ds1
df["Bagging"]=clfb1 
df["LDA"]=lda1  
df.to_csv("Check_other.csv")    



