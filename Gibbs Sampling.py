# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:09:34 2019

@author: Albert
"""
from scipy import stats
import numpy as np
from numpy import linalg as LA
def uno2p3(y,chain):
    n=y.shape[0]
    k=y.shape[1]
    a=np.ones((1,k))*2
    b=np.ones((1,k))
    #b=-stats.norm.ppf(np.sum(y,axis=0,keepdims=True)/n)*np.sqrt(5)
    th=np.zeros((1,n))
    mu=0
    var=1
    agu=np.zeros((2,1))
    ags=np.identity(2)
    alpha=np.zeros((chain,k))
    beta=np.copy(alpha)
    theta=np.zeros((chain,n))
    
    for i in range(chain):
        ###updating z
        lp=np.dot(th.T,a)-np.dot(np.ones((n,1)),b)
        bb=stats.norm.cdf(0,lp,1)
        u=np.random.uniform(0,1,(n,k))
        t1=(bb*(1-y)+(1-bb)*y)*u+bb*y
        z=stats.norm.ppf(t1,lp,1)   ####correct
        
        ###updating theta
        v=np.sum(a**2)
        pv=1/(v+1/var)
        for j in range(n):
            mn=(np.sum(a*(z[j,:]+b))+mu/var)*pv   
            #th[0,j]=np.random.normal(mn,np.sqrt(pv),1)  ####correct
            th[0,j]=np.random.randn()*np.sqrt(pv)+mn
            
        ###udating alpha,beta
        
        x=np.vstack((th,-np.ones((1,n)))).T
        agvar=LA.inv(np.dot(x.T,x)+LA.inv(ags))
        amt=LA.cholesky(agvar)
        #bz=np.dot(agvar,(np.dot(x.T,z)+np.dot(np.dot(LA.inv(ags),agu),np.ones((1,k)))))
        bz=np.dot(agvar,(np.dot(x.T,z)+np.dot(np.dot(LA.inv(ags),agu),np.ones((1,k)))))
        #amt=LA.cholesky(LA.inv(np.dot(x.T,x)))
        #bz=np.dot(LA.inv(np.dot(x.T,x)),np.dot(x.T,z))
        for j in range(k):
            a[0,j]=0
            ttt=0
            while a[0,j]<=0:
                #t2=np.dot(amt,np.random.normal(0,1,(2,1)))+bz[:,j].reshape(2,1)
                t2=np.dot(amt,np.random.randn(2,1))+bz[:,j].reshape(2,1)
                a[0,j]=t2[0,0]
                b[0,j]=t2[1,0]
                ttt=ttt+1
                if(ttt>500 and a[0,j]<=0):
                    a[0,j]=0.00001
                    
                
            
           
                
        alpha[i,:]=np.copy(a)
        beta[i,:]=np.copy(b)
        theta[i,:]=np.copy(th)
        
        print("finish", i)
        
        
    burnin=chain*(0.5)
    ae=np.copy(alpha[np.int(burnin):chain,:])
    be=np.copy(beta[np.int(burnin):chain,:])
    the=np.copy(theta[np.int(burnin):chain,:])
    return ae,be,the
#print(np.average(ae,axis=0))