#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 09:19:26 2023

@author: john
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

#6, 7, 8 in primary run, 0, 1, 2 from eviewfigssmall for transparent scatter match
with open("./reviewfigssmall/human_multivariate_Gaussian_pickles/MultivariateGaussianResults0.p", "rb") as input_file:
    results0 = pickle.load(input_file)
    
with open("./reviewfigssmall/human_multivariate_Gaussian_pickles/MultivariateGaussianResults1.p", "rb") as input_file:
    results1 = pickle.load(input_file)
    
with open("./reviewfigssmall/human_multivariate_Gaussian_pickles/MultivariateGaussianResults2.p", "rb") as input_file:
    results2 = pickle.load(input_file)
    
peak_val_index = 0
    
#join peak vals
peak_vals_all = results0[peak_val_index]
peak_vals_all.extend(results1[peak_val_index])
peak_vals_all.extend(results2[peak_val_index])

sc_above_rodent = ((np.asarray(peak_vals_all)>0.6433691320170686).sum())/len(peak_vals_all)
sc_above_human = ((np.asarray(peak_vals_all)>0.7054327142412522).sum())/len(peak_vals_all)
print("sc_above_rodent="+str(sc_above_rodent))
print("sc_above_human="+str(sc_above_human))

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(peak_vals_all, bins=100)
#plt.title("Peak Silhouette Coefficient")
plt.axvline(x=0.6433691320170686, color='#55c666ff', ymax=0.9) #stand in for rodent line (85, 198, 102)
plt.axvline(x=0.7054327142412522, color='#a099c5ff', ymax=0.9) #stand in for human line
f.set_size_inches((2.56, 2.06)) #for big-small plot
f.set_dpi(1200)
f.tight_layout()
plt.savefig("./reviewfigssmall/human_multivariate_gaussian_histogram.svg", transparent=True)