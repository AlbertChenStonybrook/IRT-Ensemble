# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:33:56 2019

@author: Albert
"""

"""
Created on Thu Oct 18 10:08:10 2018

@author: Albert
"""
# coding=utf-8

from scipy import optimize
import numpy as np
from scipy import special as sp
from scipy import stats as st
import tensorflow as tf
from scipy.stats import norm

"""
#sp.digamma(6)
#a=np.array([1,2,3;4,5,6])#alpha1=np.array([0.64,0.81,0.96,0.97,1.05,0.83,0.61,0.8,1.17,1.51]).reshape(1,10)
#alpha1=np.array([0.34,0.71,4.96,2.97,3.06,3.81,1.61,0.87,1.67,9.51]).reshape(1,10)
alpha1=np.array([0.64,0.81,0.96,0.97,1.05,0.83,0.61,0.8,1.17,1.51]).reshape(1,10)
#phi1=np.array([-1.62,-1.53,-1.29,-1.06,-0.24,-0.26,0.02,0.21,-0.67,0.48]).reshape(1,10)
phi1=alpha1
mean1=np.mean(alpha1)
phi1=phi1-np.mean(phi1)
theta1=np.random.normal(0,1,(1,1000)).reshape(1,1000)
M=np.exp((theta1.T-phi1)/2)
N=np.exp((theta1.T-phi1)*(-0.5))
n,m=M.shape   
P=st.beta.rvs(M,N)
y=st.bernoulli.rvs(P)
"""



def pre3(Y):
    n,m=Y.shape
    theta1=(np.sum(Y,1)/n-np.mean(np.sum(Y,1)/n)).reshape(1,n)
    #theta1=np.zeros((1,n))
    beta1=((1-np.sum(Y,0)/n)-np.mean(1-np.sum(Y,0)/n)).reshape(1,m)
    #print(beta1)
    #beta1=np.ones((1,m))*3
    #beta1=-1/norm.cdf(np.sum(Y,0)/n,0)
    M=np.exp((theta1.T-beta1)/2)
    N=np.exp((theta1.T-beta1)*(-0.5))
    return n,m,M,N

def EMmodel(Y,n,m,M,N,learning_rate,alpha1=0,num_iter_update=20):
    theta1=np.sum(Y,1)/n-np.mean(np.sum(Y,1)/n)
    beta1=(1-np.sum(Y,0)/n)-np.mean(1-np.sum(Y,0)/n)
    phi1=beta1.reshape(1,m)
    Mcoefficientold=sp.digamma(M+Y)-sp.digamma(N+M+1)
    Ncoefficientold=sp.digamma(N-Y+1)-sp.digamma(N+M+1)
    
    l1=np.sum(M*Mcoefficientold+N*Ncoefficientold)
    ##tensorflow
    #theta1=tf.Variable(np.random.randn(n,1).astype(np.float32))
    #phi11=tf.Variable(np.random.randn(m-1,1).astype(np.float32))
    theta1=tf.Variable(theta1.astype(np.float32))
    phi11=tf.Variable(phi1[:,0:m-1].astype(np.float32))
    Alpha1=tf.Variable(np.ones((1,m-1)).astype(np.float32))
    Alpha2=tf.Variable(np.ones(1).astype(np.float32))
    #phi1=tf.concat([phi11,tf.subtract(tf.constant([0],dtype=tf.float32,shape=[1,1]),tf.reduce_sum(phi11))],1)
    M1=tf.transpose(tf.exp((theta1-tf.transpose(phi11))*(tf.transpose(Alpha1)*tf.ones((1,n)))*0.5))
    N1=tf.transpose(tf.exp((-theta1+tf.transpose(phi11))*(tf.transpose(Alpha1)*tf.ones((1,n)))*0.5))
    Moldcoeff=tf.constant(Mcoefficientold,dtype=tf.float32)
    Noldcoeff=tf.constant(Ncoefficientold,dtype=tf.float32)
    alpha = tf.constant(alpha1,dtype=tf.float32)
    regphi = alpha*(tf.reduce_sum(phi11**2)+(tf.constant(0,dtype=tf.float32)-tf.reduce_sum(phi11))**2)
    regtheta= alpha*tf.reduce_sum(theta1**2)
    M2=tf.exp((tf.transpose(theta1)-tf.multiply(tf.ones((n,1)),tf.constant([0],dtype=tf.float32,shape=[1,1])-tf.reduce_sum(phi11)))*Alpha2*0.5)
    N2=tf.exp((tf.transpose(theta1)-tf.multiply(tf.ones((n,1)),tf.constant([0],dtype=tf.float32,shape=[1,1])-tf.reduce_sum(phi11)))*Alpha2*(-0.5))
    
    #optimizer=tf.train.AdamOptimizer(0.1)
    optimizer=tf.train.AdamOptimizer(learning_rate)
    loss=tf.reduce_sum(-tf.multiply(M1,Moldcoeff[:,0:m-1])-tf.multiply(N1,Noldcoeff[:,0:m-1]))+tf.reduce_sum(-tf.multiply(N2,tf.reshape(Noldcoeff[:,m-1],[n,1]))-tf.multiply(M2,tf.reshape(Moldcoeff[:,m-1],[n,1])))+regphi+regtheta
    
    """
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)
    """
    
    train_step = optimizer.minimize(loss)
    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    phiold=np.zeros((1,m-1))
    for e in range(3000):
        sess.run(train_step)
        if((e+1)%num_iter_update==0):
            #loss1=sess.run(loss)
            sess.run(loss)
            #theta1=tf.clip_by_value(theta1,-1,1)
            #phi11=tf.clip_by_value(phi11,-2,2)
            phi2=sess.run(phi11)
            if(np.max(np.abs(np.squeeze(phi2-phiold)))<0.005):
                print("converge")
                break  
            print("iter",e,"max difference",np.max(np.abs(np.squeeze(phi2-phiold))))             
            theta2=sess.run(theta1)
            #l=-tf.reduce_sum(tf.multiply(tf.exp((tf.transpose(theta2)-phi2)*0.5),Moldcoeff)-tf.multiply(tf.exp((-tf.transpose(theta2)+phi2)*0.5),Noldcoeff))
            #l1=sess.run(l)
            #M1=tf.transpose(tf.exp((theta2*tf.transpose(Alpha1)-tf.transpose(phi2)*Alpha1)*0.5))
            #N1=tf.transpose(tf.exp((-theta2*tf.transpose(Alpha1)+tf.transpose(phi2)*Alpha1)*0.5))
            #M2=tf.exp((tf.transpose(theta1)-tf.multiply(tf.ones((n,1)),tf.constant([0],dtype=tf.float32,shape=[1,1])-tf.reduce_sum(phi11)))*Alpha2*0.5)
            #N2=tf.exp((tf.transpose(theta1)-tf.multiply(tf.ones((n,1)),tf.constant([0],dtype=tf.float32,shape=[1,1])-tf.reduce_sum(phi11)))*Alpha2*(-0.5))
            #loss2=tf.reduce_sum(-tf.multiply(M2,Moldcoeff[:,0:m-1])-tf.multiply(N2,Noldcoeff[:,0:m-1]))+tf.reduce_sum(-tf.multiply(N22,tf.reshape(Noldcoeff[:,m-1],[n,1]))-tf.multiply(M22,tf.reshape(Moldcoeff[:,m-1],[n,1])))
            phiold=phi2
            #print(e,phi2)
            #st.pearsonr(be1,np.squeeze(phi1))
            
    sess.close()
    theta1=theta2
    beta1=phi2
    return theta1, beta1


def EMIRT(y,learning_rate=0.1,alpha=0.5):
    n,m,M,N=pre3(y)
    the,be=EMmodel(y,n,m,M,N,learning_rate=0.1,alpha1=alpha,num_iter_update=10)
    #be1=np.append(be,-np.mean(be))
    #st.pearsonr(be1,np.squeeze(phi1))
    the=np.squeeze(the)
    return the

#the=EMIRT(y)    
    