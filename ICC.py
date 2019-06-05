# -*- coding: utf-8 -*-
"""
Created on Mon May 13 17:16:18 2019

@author: Albert
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import beta


def ICC(alpha,low,high,betam):
    alpha=alpha
    delta=0.01
    low=low
    high=high
    theta=np.arange(low,high,delta)
    beta1=betam
    P=1/(1+np.exp(-alpha*theta+beta1))
    return P

low=-6
high=6
alpha=0.8
P_21=ICC(alpha,low,high,-0.6)
P_22=ICC(alpha,low,high,-0.3)
P_23=ICC(alpha,low,high,0)
P_24=ICC(alpha,low,high,0.3)
P_25=ICC(alpha,low,high,0.6)
theta=np.arange(low,high,0.01)


fig=plt.figure(figsize=(26, 26))
ax1=fig.add_subplot(131)
ax1.plot(theta,P_21)
ax1.plot(theta,P_22)
ax1.plot(theta,P_23)
ax1.plot(theta,P_24)
ax1.plot(theta,P_25)
#ax1.legend(fontsize=26)
ax1.set_xlabel("Ability", fontsize=29)
ax1.set_ylabel("Expected response function", fontsize=29)
ax1.set_title('Alpha=1',fontsize=36)
plt.xticks(size=23)
plt.yticks(size=23)
plt.legend(labels=['beta=-0.6','beta=-0.3','beta=0','beta=0.3','beta=0.6'],fontsize=26)
#plt.show()


low=-6
high=6
alpha=1.6
P_21=ICC(alpha,low,high,-0.6)
P_22=ICC(alpha,low,high,-0.3)
P_23=ICC(alpha,low,high,0)
P_24=ICC(alpha,low,high,0.3)
P_25=ICC(alpha,low,high,0.6)
theta=np.arange(low,high,0.01)

ax2=fig.add_subplot(132)
ax2.plot(theta,P_21)
ax2.plot(theta,P_22)
ax2.plot(theta,P_23)
ax2.plot(theta,P_24)
ax2.plot(theta,P_25)
#ax1.legend(fontsize=26)
ax2.set_xlabel("Ability", fontsize=29)
ax2.set_ylabel("Expected response function", fontsize=29)
ax2.set_title('Alpha=2',fontsize=36)
plt.xticks(size=23)
plt.yticks(size=23)
plt.legend(labels=['beta=-0.6','beta=-0.3','beta=0','beta=0.3','beta=0.6'],fontsize=26)



low=-6
high=6
alpha=3.2
P_21=ICC(alpha,low,high,-0.6)
P_22=ICC(alpha,low,high,-0.3)
P_23=ICC(alpha,low,high,0)
P_24=ICC(alpha,low,high,0.3)
P_25=ICC(alpha,low,high,0.6)
theta=np.arange(low,high,0.01)

ax3=fig.add_subplot(133)
ax3.plot(theta,P_21)
ax3.plot(theta,P_22)
ax3.plot(theta,P_23)
ax3.plot(theta,P_24)
ax3.plot(theta,P_25)
#ax1.legend(fontsize=26)
ax3.set_xlabel("Ability", fontsize=29)
ax3.set_ylabel("Expected response function", fontsize=29)
ax3.set_title('Alpha=3',fontsize=36)
plt.xticks(size=23)
plt.yticks(size=23)
plt.legend(labels=['beta=-0.6','beta=-0.3','beta=0','beta=0.3','beta=0.6'],fontsize=26)










plt.show()