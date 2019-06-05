# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:15:08 2019

@author: Albert
"""

import numpy as np
import matplotlib.pyplot as plt

a1=np.array([-1.024,-0.934,-0.694,-0.464,0.356,0.336,0.616,0.806,-0.074,1.076])
my11=np.array([-0.94,-0.99,-0.732,-0.56,0.472,0.336,0.694,0.85,-0.171,1.23])
my12=np.array([-0.953,-0.91,-0.709,-0.514,0.408,0.319,0.576,0.757,-0.145,1.141])
my13=np.array([-1.107,-1.08,-0.869,-0.579,0.503,0.382,0.722,0.956,-0.172,1.243])

avgmse1=np.sum((a1-my11)**2+(a1-my12)**2+(a1-my13)**2)/3
"""
r11=np.abs(my11-a1)/np.abs(a1)
r12=np.abs(my12-a1)/np.abs(a1)
r13=np.abs(my13-a1)/np.abs(a1)
"""
r11=(a1-my11)**2/avgmse1
r12=(a1-my12)**2/avgmse1
r13=(a1-my13)**2/avgmse1

a2=np.array([-9.075,-2.485,-3.395,-1.015,-0.075,0.165,4.035,6.485,-6.395,11.755])
my21=np.array([-4.35,-1.3,-1.2,-2.2,0.461,1.36,3.194,3.85,-3.171,3.23])
my22=np.array([-3.256,-1.296,-2.24,-0.192,1.593,0.96,3.045,3.783,-2.904,3.534])
my23=np.array([-9.824,-2.68,-3.85,-0.203,1.248,1.628,5.591,6.998,-8.462,9.553])

avgmse2=np.sum((a2-my21)**2+(a2-my22)**2+(a2-my23)**2)/3
"""
r21=np.abs(my21-a2)/np.abs(a2)
r22=np.abs(my22-a2)/np.abs(a2)
r23=np.abs(my23-a2)/np.abs(a2)
"""
r21=(a2-my21)**2/avgmse2
r22=(a2-my22)**2/avgmse2
r23=(a2-my23)**2/avgmse2

fig=plt.figure(figsize=(26, 26))
ind = np.arange(len(r11))  # the x locations for the groups
width = 0.35  # the width of the bars
ax1=fig.add_subplot(211)
rects11 = ax1.bar(ind - width/2, r11, width, color='SkyBlue', label='Model 1')
rects21 = ax1.bar(ind + width/2, r12, width,color='IndianRed', label='Model 2')
rects31 = ax1.bar(ind + width, r13, width, color='Green', label='Model 3')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('Ratio',fontsize=26)
ax1.set_title('Dataset 1',fontsize=36)
plt.xticks(ind,('Beta1', 'Beta2', 'Beta3', 'Beta4', 'Beta5','Beta6','Beta7','Beta8','Beta9','Beta10'))
ax1.legend(fontsize=19)
plt.ylim((0, 1))
plt.xticks(size=23)
plt.yticks(size=23)

ax2=fig.add_subplot(212)
rects21 = ax2.bar(ind - width/2, r21, width, color='SkyBlue', label='Model 1')
rects22 = ax2.bar(ind + width/2, r22, width,color='IndianRed', label='Model 2')
rects23 = ax2.bar(ind + width, r23, width, color='Green', label='Model 3')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax2.set_ylabel('Ratio',fontsize=26)
ax2.set_title('Dataset 2',fontsize=36)
plt.xticks(ind,('Beta1', 'Beta2', 'Beta3', 'Beta4', 'Beta5','Beta6','Beta7','Beta8','Beta9','Beta10'))
ax2.legend(fontsize=19)
plt.ylim((0, 1))
plt.xticks(size=23)
plt.yticks(size=23)
plt.show()
