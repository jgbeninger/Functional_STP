#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:14:17 2023

@author: john
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

#saved fits 3,4,5 are for the original 1000 each test of facilitation metric
with open("./reviewfigs/refit_pickles/model_params3.p", "rb") as input_file:
    model3 = pickle.load(input_file)
    #wider 21
    
with open("./reviewfigs/refit_pickles/model_params4.p", "rb") as input_file:
    model4 = pickle.load(input_file)
    #wider 22
    
with open("./reviewfigs/refit_pickles/model_params5.p", "rb") as input_file:
    model5 = pickle.load(input_file)
    #wider 23
    
#------------------------------------------------------------------------------
#define facilitation metric
def fac_met(mu_amps, mu_taus):
    result = 0
    for i in range(0, len(mu_amps)):
        result += mu_amps[i]*mu_taus[i]
    return result

#------------------------------------------------------------------------------

#regular
"""
mu_taus = [15, 200, 300]
pre_fac_mets = []

#calculate facilitation metrics pre-fitting
for file in [model3[0], model4[0], model5[0]]:
    for run in file:
        for params in run:
            baseline, mu_amp1, mu_amp2, mu_amp3, SD = params
            mu_amps = (mu_amp1, mu_amp2, mu_amp3)
            pre_fac_mets.append(fac_met(mu_amps, mu_taus))


"""
#faciliation shift testing
mu_taus = [15, 200, 300]
pre_fac_mets = []

#calculate facilitation metrics pre-fitting
for file in [model3[0], model4[0], model5[0]]:
    for run in file:
        for params in run:
            baseline, mu_amp1, mu_amp2, mu_amp3, SD = params
            mu_amps = (mu_amp1, mu_amp2, mu_amp3)
            pre_fac_mets.append(fac_met(mu_amps, mu_taus))


#plot pre histogram
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(pre_fac_mets, bins=50)
plt.title("Facilitation Measures Before Refitting")
plt.show()

post_fac_mets = []

#regular
"""
#calculate facilitation metrics pre-fitting
for file in [model3[1], model4[1], model5[1]]:
    for run in file:
        for params in run:
            baseline, mu_amp1, mu_amp2, mu_amp3, SD = params
            mu_amps = (mu_amp1, mu_amp2, mu_amp3)
            post_fac_mets.append(fac_met(mu_amps, mu_taus))
"""

#calculate facilitation metrics post-fitting
for file in [model3[1], model4[1], model5[1]]:
    for run in file:
        for params in run:
            baseline, mu_amp1, mu_amp2, mu_amp3, SD = params
            mu_amps = (mu_amp1, mu_amp2, mu_amp3)
            post_fac_mets.append(fac_met(mu_amps, mu_taus))


#plot post histogram
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(post_fac_mets, bins=50)
plt.title("Facilitation Measures After Refitting")
plt.show()

#calculate bias by param            
models = [model3, model4, model5]

biases = [[] for i in range(0, 5)]
values = [[] for i in range(0, 5)]
for i in range(0, 3):
    model_pre = models[i][0]
    model_post = models[i][1]
    for j in range(0, len(model_pre)):
        runs_pre = model_pre[j]
        runs_post = model_post[j]
        for k in range(0, len(runs_pre)):
            params_pre = runs_pre[k]
            params_post = runs_post[k]
            for l in range(0, 5):
                biases[l].append(params_post[l]-params_pre[l])
                values[l].append(params_post[l])

for i in range(0, len(biases)): 
    param_biases = biases[i]
    f =plt.figure()
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.hist(param_biases, bins=50)
    plt.title("Bias in param"+str(i))
    plt.show()
    
print("len(post_fac_mets)"+str(len(post_fac_mets)))
print("len(values[0])"+str(len(values[0])))
for i in range(0, len(values)): 
    param_values = values[i]
    f =plt.figure()
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.hist(param_values , bins=50)
    plt.title("Values of param"+str(i))
    plt.show()
    
print("mean pre fracilitation metric:"+str(np.mean(pre_fac_mets)))

print("mean post_facilitation metric:"+str(np.mean(post_fac_mets)))