# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:15:36 2019

@author: Albert



"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
#from StanIRT import MCMC
#from sklearn.datasets import load_digits
from Mymodel1 import Performancematrix
from Mymodel1 import plotk,normcdf,plotm
from sklearn import preprocessing
from Mymodel1 import Performancematrix
from Mymodel1 import plotk,normcdf,uni,plotm
from sklearn import preprocessing
from uno2p import uno2p2,uno2p1,uno2p3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from scipy import stats







from scipy import stats
stats.ttest_ind(rvs1,rvs2)
###model1

def makeprediction(pre):
    l=np.zeros(pre.shape[0])
    for i in range(pre.shape[0]):
        
        if(pre[i]>0.5):
            l[i]=1
        else:
            l[i]=0
    return l;


Bld=pd.read_csv("Bld.csv",header=None)
Bld=Bld.dropna()
y=Bld.iloc[:,6]
y=y-1
X=Bld.drop(6,axis=1)
X=np.array(X)
X=preprocessing.scale(X)

#### boxplot

Bld=pd.read_csv("Bld.csv",header=None)
Bld=Bld.dropna()
y=Bld.iloc[:,6]
y=y-1
X=Bld.drop(6,axis=1)
X=np.array(X)
X=preprocessing.scale(X)
ntree1=[2,16,32,64]



t_statbld=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=5,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statbld[nn,0]=t1
    t_statbld[nn,1]=t2
    t_statbld[nn,2]=t3
    t_statbld[nn,3]=t4
    t_statbld[nn,4]=t5
    t_statbld[nn,5]=t6
    print("finish tree",n)
    
    nn=nn+1
        
np.savetxt("report\Bldstat.csv",t_statbld)  


for i in range(7):
    np.savetxt("report\Bldacctotal"+str(i)+".csv",acc_total[i,:,:])
    
    
###IRIS

ntree1=[128,256,512,1024]
IRIS=load_iris()
X=IRIS['data']
y=IRIS['target']
X=preprocessing.scale(X)
accuracy1=np.empty(0)
accuracy2=np.empty(0)
    

t_statiris=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=5,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
        C,T=Performancematrix(Xtrain,Xtest,ytrain,ntree)
        C=C.astype(int)
        C=C.T
        #C=C.T
        #P=(np.repeat(30,C.shape[0])-np.sum(C,axis=1))*3
        #data1={"np":ntree,"ni":Xtrain.shape[0],"U":C,"prior":P}
        #data1={"np":ntree,"ni":Xtrain.shape[0],"U":C}
        #fit = sm.sampling(data=data1, iter=300, chains=1,n_jobs=1)
        ae,be,the=uno2p3(C,1000)
        #a=np.zeros(3)
        
        #ae,be,the=uno2p3(C,2000)
        #ae,be,the=uno3p1(C,2000)
        #l=np.average(the,axis=0)
        l=np.average(the,axis=0)
        l=normcdf(l)
        w=l/np.sum(l)
        a=np.zeros(3)
        pre=np.zeros(Xtest.shape[0])
        for j in range(T.shape[0]):
            t=T[j,:]
            a[0]=np.sum(w[t==0])
            a[1]=np.sum(w[t==1])
            a[2]=np.sum(w[t==2])
            pre[j]=np.argmax(a)
        l=np.abs(pre-np.squeeze(ytest))
        l[l>0]=1
        a2=1-np.sum(l)/Xtest.shape[0]
        accuracy2=np.append(accuracy2,a2)
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statiris[nn,0]=t1
    t_statiris[nn,1]=t2
    t_statiris[nn,2]=t3
    t_statiris[nn,3]=t4
    t_statiris[nn,4]=t5
    t_statiris[nn,5]=t6
    print("finish tree",n)
    
    nn=nn+1
        
np.savetxt("report\IRISstat.csv",t_statiris)  
        
  
for i in range(7):
    np.savetxt("report\irisacctotal"+str(i)+".csv",acc_total[i,:,:])
    
#ECOLI


ntree1=[128,256,512,1024]
ECOLI=pd.read_csv("ECOLI.csv",header=None)
ECOLI=ECOLI.dropna()
Choice_mapping = {'cp':0,'im':1,'pp':2,'imU':3,'om':4,'omL':5,'imL':6,'imS':7}
y=ECOLI.iloc[:,7].map(Choice_mapping)
X=ECOLI.drop(7,axis=1)
X=np.array(X)
X=preprocessing.scale(X)
    

t_statiris=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=3,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        l=normcdf(l)
        w=l/np.sum(l)
        a=np.zeros(8)
        pre=np.zeros(Xtest.shape[0])
        for j in range(T.shape[0]):
            t=T[j,:]
            a[0]=np.sum(w[t==0])
            a[1]=np.sum(w[t==1])
            a[2]=np.sum(w[t==2])
            a[3]=np.sum(w[t==3])
            a[4]=np.sum(w[t==4])
            a[5]=np.sum(w[t==5])
            a[6]=np.sum(w[t==6])
            a[7]=np.sum(w[t==7])
            pre[j]=np.argmax(a)
        l=np.abs(pre-np.squeeze(ytest))
        l[l>0]=1
        a2=1-np.sum(l)/Xtest.shape[0]
        accuracy2=np.append(accuracy2,a2)
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statiris[nn,0]=t1
    t_statiris[nn,1]=t2
    t_statiris[nn,2]=t3
    t_statiris[nn,3]=t4
    t_statiris[nn,4]=t5
    t_statiris[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
    
np.savetxt("report\ECOLIstat1.csv",t_statiris)  
    
for i in range(4):
    np.savetxt("report\Ecolisacctotal"+str(i)+".csv",acc_total[i,:,:])    

##IPLD


ILPD=pd.read_csv("ILPD.csv",header=None)
ILPD=ILPD.dropna()
Choice_mapping = {'Female':0,'Male':1}
X=ILPD
X.iloc[:,1]=X.iloc[:,1].map(Choice_mapping )
y=X.iloc[:,10]
y=y-1
X=X.drop(10,axis=1)
X.iloc[:,0]=preprocessing.scale(X.iloc[:,0])
X.iloc[:,2]=preprocessing.scale(X.iloc[:,2])
X.iloc[:,3]=preprocessing.scale(X.iloc[:,3])
X.iloc[:,4]=preprocessing.scale(X.iloc[:,4])
X.iloc[:,5]=preprocessing.scale(X.iloc[:,5])
X.iloc[:,6]=preprocessing.scale(X.iloc[:,6])
X.iloc[:,7]=preprocessing.scale(X.iloc[:,7])
X.iloc[:,8]=preprocessing.scale(X.iloc[:,8])
X.iloc[:,9]=preprocessing.scale(X.iloc[:,9])
X=pd.get_dummies(X,columns=[1])
X=pd.DataFrame(X,dtype=np.float)
X=np.array(X)


t_statILPD=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=3,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statILPD[nn,0]=t1
    t_statILPD[nn,1]=t2
    t_statILPD[nn,2]=t3
    t_statILPD[nn,3]=t4
    t_statILPD[nn,4]=t5
    t_statILPD[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
    
        
np.savetxt("report\IPLDstat.csv",t_statILPD)  

for i in range(7):
    np.savetxt("report\IPLDisacctotal"+str(i)+".csv",acc_total[i,:,:])  
    
    
    
###TAE



TAE=pd.read_csv("TAE.csv",header=None)
TAE=TAE.dropna()
y=TAE.iloc[:,5]
y=y-1
X=TAE.drop(5,axis=1)
X.iloc[:,4]=preprocessing.scale(X.iloc[:,4])
X=pd.get_dummies(X,columns=[1,2])
X=pd.DataFrame(X,dtype=np.float)
X=np.array(X)

ntree1=[128,256,512,1024]
t_statTAE=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=3,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        l=normcdf(l)
        w=l/np.sum(l)
        a=np.zeros(3)
        pre=np.zeros(Xtest.shape[0])
        for j in range(T.shape[0]):
            t=T[j,:]
            a[0]=np.sum(w[t==0])
            a[1]=np.sum(w[t==1])
            a[2]=np.sum(w[t==2])
            pre[j]=np.argmax(a)
        l=np.abs(pre-np.squeeze(ytest))
        l[l>0]=1
        a2=1-np.sum(l)/Xtest.shape[0]
        accuracy2=np.append(accuracy2,a2)
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statTAE[nn,0]=t1
    t_statTAE[nn,1]=t2
    t_statTAE[nn,2]=t3
    t_statTAE[nn,3]=t4
    t_statTAE[nn,4]=t5
    t_statTAE[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
    
        
np.savetxt("report\TAEstat.csv",t_statTAE)  



    
##LEN
ntree1=[64,128,256,512]
len1=pd.read_csv("len.csv",header=None)
len1=len1.dropna()
y=len1.iloc[:,4]
y=y-1
X=len1.drop(4,axis=1)
X=pd.get_dummies(X,columns=[0,1,2,3])
X=np.array(X)


t_statLEN=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=5,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        l=normcdf(l)
        w=l/np.sum(l)
        a=np.zeros(3)
        pre=np.zeros(Xtest.shape[0])
        for j in range(T.shape[0]):
            t=T[j,:]
            a[0]=np.sum(w[t==0])
            a[1]=np.sum(w[t==1])
            a[2]=np.sum(w[t==2])
            pre[j]=np.argmax(a)
        l=np.abs(pre-np.squeeze(ytest))
        l[l>0]=1
        a2=1-np.sum(l)/Xtest.shape[0]
        accuracy2=np.append(accuracy2,a2)
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statLEN[nn,0]=t1
    t_statLEN[nn,1]=t2
    t_statLEN[nn,2]=t3
    t_statLEN[nn,3]=t4
    t_statLEN[nn,4]=t5
    t_statLEN[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
        
np.savetxt("report\LENstat.csv",t_statLEN)  
    
for i in range(6):
    np.savetxt("report\LENisacctotal"+str(i)+".csv",acc_total[i,:,:]) 
    
###AUS

AUS=pd.read_csv("AUS.csv",header=None)
y=AUS.iloc[:,14]
X=AUS.drop(14,axis=1)
X.iloc[:,1]=preprocessing.scale(X.iloc[:,1])
X.iloc[:,2]=preprocessing.scale(X.iloc[:,2])
X.iloc[:,6]=preprocessing.scale(X.iloc[:,6])
X.iloc[:,9]=preprocessing.scale(X.iloc[:,9])
X=pd.get_dummies(X,columns=[0,3,4,5,7,8,10,11])
X=pd.DataFrame(X,dtype=np.float)
X=np.array(X)

ntree1=[128,256,512,1024]
ntree2=[2048]
t_statAUS=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
t_statAUS=np.zeros((1,6))
acc_total=np.zeros((1,7,10))
nn=0
for n in ntree2:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=5,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        l=normcdf(l)
        w=l/np.sum(l)
        a=np.zeros(4)
        pre=np.zeros(Xtest.shape[0])
        for j in range(T.shape[0]):
            t=T[j,:]
            a[0]=np.sum(w[t==0])
            a[1]=np.sum(w[t==1])
            a[2]=np.sum(w[t==2])
            a[3]=np.sum(w[t==3])
            pre[j]=np.argmax(a)
        l=np.abs(pre-np.squeeze(ytest))
        l[l>0]=1
        a2=1-np.sum(l)/Xtest.shape[0]
        accuracy2=np.append(accuracy2,a2)
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statAUS[nn,0]=t1
    t_statAUS[nn,1]=t2
    t_statAUS[nn,2]=t3
    t_statAUS[nn,3]=t4
    t_statAUS[nn,4]=t5
    t_statAUS[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
        
np.savetxt("report\AUSstat1.csv",t_statAUS)  
    
for i in range(6):
    np.savetxt("report\AUSisacctotal"+str(i)+".csv",acc_total[i,:,:]) 
    
###SPE

def makeprediction(pre):
    l=np.zeros(pre.shape[0])
    for i in range(pre.shape[0]):
        
        if(pre[i]>0.5):
            l[i]=1
        else:
            l[i]=0
    return l;


SPE=pd.read_csv("SPE.csv",header=None)
SPE=SPE.dropna()
y=SPE.iloc[:,0]
X=SPE.drop(0,axis=1)
X=np.array(X)
X=preprocessing.scale(X)

ntree1=[128,256,512,1024]
t_statSPE=np.zeros((4,6))
acc_total=np.zeros((4,7,10))
nn=0
for n in ntree1:
    accuracy1=np.empty(0)
    accuracy2=np.empty(0)
    accuracy3=np.empty(0)
    accuracy4=np.empty(0)
    accuracy5=np.empty(0)
    accuracy6=np.empty(0)
    accuracy7=np.empty(0)
    for i in range(10):   
        Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.3,random_state=i)
        forest=RandomForestClassifier(n_estimators=n,max_depth=5,max_features='sqrt')
        forest.fit(Xtrain,ytrain)
        a1=forest.score(Xtest,ytest)
        accuracy1=np.append(accuracy1,a1)
        ntree=n
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
        gbr = GradientBoostingClassifier(n_estimators=ntree, max_depth=2, max_features='sqrt',learning_rate=0.8,random_state=i).fit(Xtrain,ytrain)
        sv = svm.LinearSVC().fit(Xtrain,ytrain)
        ds=DecisionTreeClassifier(max_depth=3,random_state=i).fit(Xtrain,ytrain)
        clfb = BaggingClassifier(base_estimator= DecisionTreeClassifier(max_depth=3),n_estimators=ntree,max_samples=0.5,max_features=0.5,random_state=i).fit(Xtrain,ytrain)
        lda = LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.9).fit(Xtrain,ytrain)
        acc_gbr=gbr.score(Xtest,ytest)
        accuracy3=np.append(accuracy3,acc_gbr)
        acc_sv=sv.score(Xtest,ytest)
        accuracy4=np.append(accuracy4,acc_sv)
        acc_ds=ds.score(Xtest,ytest)
        accuracy5=np.append(accuracy5,acc_ds)
        acc_clfb=clfb.score(Xtest,ytest)
        accuracy6=np.append(accuracy6,acc_clfb)
        acc_lda=lda.score(Xtest,ytest)
        accuracy7=np.append(accuracy7,acc_lda)
        print("finish_inner",n,"iter",i)
    acc_total[nn,0,:]=accuracy1
    acc_total[nn,1,:]=accuracy2
    acc_total[nn,2,:]=accuracy3
    acc_total[nn,3,:]=accuracy4
    acc_total[nn,4,:]=accuracy5
    acc_total[nn,5,:]=accuracy6
    acc_total[nn,6,:]=accuracy7
    t1,s1=stats.ttest_ind(accuracy2,accuracy1)
    t2,s2=stats.ttest_ind(accuracy2,accuracy3)
    t3,s3=stats.ttest_ind(accuracy2,accuracy4)
    t4,s4=stats.ttest_ind(accuracy2,accuracy5)
    t5,s5=stats.ttest_ind(accuracy2,accuracy6)
    t6,s6=stats.ttest_ind(accuracy2,accuracy7)
    t_statSPE[nn,0]=t1
    t_statSPE[nn,1]=t2
    t_statSPE[nn,2]=t3
    t_statSPE[nn,3]=t4
    t_statSPE[nn,4]=t5
    t_statSPE[nn,5]=t6
    print("finish tree",n)
    nn=nn+1
        
np.savetxt("report\SPEstat.csv",t_statSPE)  



    

    
    
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


###rf
My1vsRF=pd.read_csv("My1vsRF.csv",header=None)
X=My1vsRF.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataRF= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataRF=dataRF.ix[:,order]
dataRF.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("Random Forest VS Mymodel1")
plt.show()

##
My1vsGBDT=pd.read_csv("My1vsGBDT.csv",header=None)
X=My1vsGBDT.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataGBDT= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataGBDT=dataGBDT.ix[:,order]
dataGBDT.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("Gradient Boosting VS Mymodel1")
plt.show()
###SVM

My1vsSVM=pd.read_csv("My1vsSVM.csv",header=None)
X=My1vsSVM.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataSVM= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataSVM=dataSVM.ix[:,order]
dataSVM.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("SVM VS Mymodel1")
plt.show()


######
My1vsDS=pd.read_csv("My1vsDS.csv",header=None)
X=My1vsDS.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataDS= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataDS=dataDS.ix[:,order]
dataDS.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("Single tree VS Mymodel1")
plt.show()

####
My1vsBag=pd.read_csv("My1vsBag.csv",header=None)
X=My1vsBag.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataBag= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataBag=dataBag.ix[:,order]
dataBag.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("Bagging VS Mymodel1")
plt.show()


###LDA
My1vsLDA=pd.read_csv("My1vsLDA.csv",header=None)
X=My1vsLDA.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataLDA= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataLDA=dataLDA.ix[:,order]
dataLDA.boxplot()
plt.ylabel("T_statistics")
plt.xlabel("Ensemble size")
plt.title("LDA VS Mymodel1")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

plt.subplots_adjust(wspace=0.3, hspace=0.7)
fig=plt.figure(figsize=(36, 36))
plt.subplot(2,3,1)
My1vsLDA=pd.read_csv("My1vsLDA.csv",header=None)
X=My1vsLDA.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataLDA= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
dataLDA=dataLDA.ix[:,order]
dataLDA.boxplot(flierprops={'marker':'x','markeredgecolor':'blue'})

plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. LDA",fontsize=50)

###SVM

plt.subplot(2,3,2)
My1vsSVM=pd.read_csv("My1vsSVM.csv",header=None)
X=My1vsSVM.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataSVM= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})    
dataSVM=dataSVM.ix[:,order]
dataSVM.boxplot(flierprops={'marker':'x','markeredgecolor':'blue'})
plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. SVM",fontsize=50)

###DS
plt.subplot(2,3,3)
My1vsDS=pd.read_csv("My1vsDS.csv",header=None)
X=My1vsDS.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataDS= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataDS=dataDS.ix[:,order]
dataDS.boxplot()
plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. Single tree",fontsize=50)

###Bagging
plt.subplot(2,3,4)
My1vsBag=pd.read_csv("My1vsBag.csv",header=None)
X=My1vsBag.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataBag= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataBag=dataBag.ix[:,order]
dataBag.boxplot()
plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. Bagging",fontsize=50)

###GBDT
plt.subplot(2,3,5)
My1vsGBDT=pd.read_csv("My1vsGBDT.csv",header=None)
X=My1vsGBDT.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataGBDT= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataGBDT=dataGBDT.ix[:,order]
dataGBDT.boxplot()
plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. Gradient Boosting",fontsize=50)

plt.subplot(2,3,6)
My1vsRF=pd.read_csv("My1vsRF.csv",header=None)
X=My1vsRF.T
X=np.array(X)
order = ['B128', 'B256', 'B512', 'B1024', 'B2048']
dataRF= pd.DataFrame({
    "B128":X[:,0],
    "B256":X[:,1],
    "B512":X[:,2],
    "B1024":X[:,3],
    "B2048":X[:,4]
})
    
dataRF=dataRF.ix[:,order]
dataRF.boxplot()
plt.ylabel("T_statistics",fontsize=39)
plt.xlabel("Ensemble size",fontsize=39)
plt.xticks(size=36)
plt.yticks(size=36)
plt.title("Model 2 vs. Random Forest",fontsize=50)
plt.show()

