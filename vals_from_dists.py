#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 11:44:24 2023

@author: john
"""

import numpy as np
import pickle 

filename = './Figures/dists/modelpostbootstrapdists1.3mMfull.p'
infile = open(filename,'rb')
post_model_dists = pickle.load(infile)
infile.close()

filename = './Figures/dists/modelprebootstrapdists1.3mMfull.p'
infile = open(filename,'rb')
pre_model_dists = pickle.load(infile)
infile.close()


filename = './Figures/dists/physprebootstrapdists1.3mMfull.p'
infile = open(filename,'rb')
pre_phys_dists = pickle.load(infile)
infile.close()

filename = './Figures/dists/physpostbootstrapdists1.3mMfull.p'
infile = open(filename,'rb')
post_phys_dists = pickle.load(infile)
infile.close()

#post

post_baselines = np.asarray(post_model_dists[1])
post_baselines_mean_acc = np.mean(post_baselines)  
post_baselines_std_acc = np.std(post_baselines) 

model_post_accuracies = np.asarray(post_model_dists[0])
model_post_mean_acc = np.mean(model_post_accuracies)  
model_post_std_acc = np.std(model_post_accuracies) 

print("post baseline = "+str(post_baselines_mean_acc ))
print("model post accuracy = "+str(model_post_mean_acc))
print("modle post std = "+str(model_post_std_acc))


phys_post_accuracies = np.asarray(post_phys_dists[0])
phys_post_mean_acc = np.mean(phys_post_accuracies)  
phys_post_std_acc = np.std(phys_post_accuracies) 

print("phys post accuracy = "+str(phys_post_mean_acc))
print("phys post std = "+str(phys_post_std_acc))

#pre

pre_baselines = np.asarray(pre_model_dists[1])
pre_baselines_mean_acc = np.mean(pre_baselines)  
pre_baselines_std_acc = np.std(pre_baselines) 

model_pre_accuracies = np.asarray(pre_model_dists[0])
model_pre_mean_acc = np.mean(model_pre_accuracies)  
model_pre_std_acc = np.std(model_pre_accuracies) 

print("pre baseline = "+str(pre_baselines_mean_acc ))
print("model pre accuracy = "+str(model_pre_mean_acc))
print("model pre std = "+str(model_pre_std_acc))


phys_pre_accuracies = np.asarray(pre_phys_dists[0])
phys_pre_mean_acc = np.mean(phys_pre_accuracies)  
phys_pre_std_acc = np.std(phys_pre_accuracies) 

print("phys pre accuracy = "+str(phys_pre_mean_acc))
print("phys prestd = "+str(phys_pre_std_acc))