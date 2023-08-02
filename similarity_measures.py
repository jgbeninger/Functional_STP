#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 13:07:25 2022

@author: john
Code to compute temporal cross-correlations between kernels as time series
"""
import statistics
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.signal import correlate
from itertools import combinations
from scipy.stats import spearmanr
from scipy.stats import pearsonr
#--------------------------------------------------

def weighted_similarity(corr_mat, SD_list, SD_SD, alpha_corr=0.75, alpha_sd=0.25):
    SDs_mat = np.empty((len(SD_list), len(SD_list)))
    for i in range(0, len(SD_list)):
        for j in range(0, len(SD_list)):
            #SDs_mat[i, j] = math.pow(math.pow((SD_list[i]-SD_list[j]), 2), 0.5)/SD_SD
            SDs_mat[i, j] = math.pow((SD_list[i]-SD_list[j]), 2)/SD_SD
    #similarities = alpha_corr*((corr_mat+1)/2) + alpha_sd*SDs_mat
    similarities = alpha_corr*((corr_mat)) - alpha_sd*SDs_mat
    return similarities

#--------------------------------------------------

#define taus for kernel
mu_kernel_taus = [15, 200, 300]

dir_name_fig6 = "./Figures/fig6/"

#pre raw
filename = 'saved_kernels/data_centers_pre_1.3mM.pickle'
infile = open(filename,'rb')
data_centers_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/raw_unique_pre_1.3mM.pickle'
infile = open(filename,'rb')
raw_pre_types = pickle.load(infile)
infile.close()

filename = 'saved_kernels/group_rows_pre_1.3mM.pickle'
infile = open(filename,'rb')
group_rows_pre = pickle.load(infile)
infile.close()

#post raw
filename = 'saved_kernels/data_centers_post_1.3mM.pickle'
infile = open(filename,'rb')
data_centers_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/raw_unique_post_1.3mM.pickle'
infile = open(filename,'rb')
raw_post_types = pickle.load(infile)
infile.close()

filename = 'saved_kernels/group_rows_post_1.3mM.pickle'
infile = open(filename,'rb')
group_rows_post= pickle.load(infile)
infile.close()


#pre supervised
filename = 'saved_kernels/clf_means_pre_1.3mM.pickle'
infile = open(filename,'rb')
clf_means_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/prediction_rows_pre_1.3mM.pickle'
infile = open(filename,'rb')
prediction_rows_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/unique_pre_1.3mM.pickle'
infile = open(filename,'rb')
unique_pre = pickle.load(infile)
infile.close()


#post supervised
filename = 'saved_kernels/clf_means_post_1.3mM.pickle'
infile= open(filename,'rb')
clf_means_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/prediction_rows_post_1.3mM.pickle'
infile = open(filename,'rb')
prediction_rows_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/unique_post_1.3mM.pickle'
infile = open(filename,'rb')
unique_post = pickle.load(infile)
infile.close()

#rodent unsupervised
filename = 'saved_kernels/OPTICS_means_post_1.3mM.pickle'
infile = open(filename,'rb')
surrogate_means =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_syn_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
syn_by_cluster_surr =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_runs_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
runs_by_cluster_surr =  pickle.load(infile)
infile.close()

#human unsupervised
filename = 'saved_kernels/OPTICSHuman_means_post_1.3mM.pickle'
infile = open(filename,'rb')
human_surrogate_means =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICSHuman_syn_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
human_syn_by_cluster_surr =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICSHuman_runs_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
human_runs_by_cluster_surr =  pickle.load(infile)
infile.close()


#SD comp
filename = 'saved_kernels/OPTICS_SDs_post_1.3mM.pickle'
infile = open(filename,'rb')
rodent_SDs = pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICSHuman_SDs_post_1.3mM.pickle'
infile = open(filename,'rb')
human_SDs = pickle.load(infile)
infile.close()

filename = 'saved_kernels/clf_SDs_pre_1.3mM.pickle'
infile = open(filename,'rb')
clf_SDs_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/clf_SDs_post_1.3mM.pickle'
infile = open(filename,'rb')
clf_SDs_post = pickle.load(infile)
infile.close()


#full model_data
filename = 'saved_kernels/model_data_1.3mM.pickle'
infile = open(filename,'rb')
rodent_model_data = pickle.load(infile)
infile.close()

filename = 'saved_kernels/Human_model_data.pickle'
infile = open(filename,'rb')
human_model_data = pickle.load(infile)
infile.close()

#calc SD of SDs
rodent_model_arr = np.asarray(rodent_model_data)
human_model_arr = np.asarray(human_model_data)
rodent_SD_of_SDs = np.std(rodent_model_arr[:,4]) #should be SD of all SD entries
human_rodent_joint_model_data = np.concatenate((rodent_model_arr, human_model_arr))
joint_SD_of_SDs = np.std(human_rodent_joint_model_data[:,4]) #should be SD of all SD entries
#--------------------------------------------------
#functions

def gen_kernel(mu_amps, mu_taus, num_timesteps, mu_baseline=None, dt=1):
    #set up timespan of 2000ms with 0.1ms time bins
    dt = 0.1  # ms per bin
    T = 2e3  # in ms
    
    t = np.arange(0, T, dt)
    spktr = np.zeros(t.shape)
    spktr[[4000]] = 1 #set one spike at 400ms
    
    kernels = [1 / tauk * np.exp(-t / tauk) for tauk in mu_taus] #generate mu kernels wrt time
    
    if mu_baseline == None:
        mu_baseline = 0 
    
    y_vals = np.roll(mu_amps[0] * kernels[0][:10000] + mu_amps[1] * kernels[1][:10000]+ mu_amps[2] * kernels[2][:10000] + mu_baseline, 2000)
    x_vals = t[:10000] - 200 
    y_vals[0:2000] = mu_baseline
    
    return (x_vals, y_vals)

#--------------------------------------------------
#get kernels as time series for all conditions 

#human unsupervised
human_series = {}
human_SDs = {}
for i in range(0, len(human_surrogate_means)):
    label = "H"+str(i)
    mu_amps = [human_surrogate_means[i,1], human_surrogate_means[i, 2], human_surrogate_means[i,3]] #add 1 to all values
    mu_baseline = human_surrogate_means[i,0] 
    print("cluster "+str(i))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    human_series[label] = kernel_y
    human_SDs[label] = human_surrogate_means[i,4] #get mean SD

#rodent unsupervised
Rodent_OPTICS_series = {}
Rodent_OPTICS_SDs = {}
for i in range(0, len(surrogate_means)):
    label = "R"+str(i)
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    print("cluster "+str(i))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    Rodent_OPTICS_series[label] = kernel_y
    Rodent_OPTICS_SDs[label] = surrogate_means[i,4]
    
#rodent mean pre
raw_pre_series = {}
raw_pre_SDs = {}
for i in range(0, len(data_centers_pre)):
    label = raw_pre_types[i]
    mu_amps = [data_centers_pre[i,1], data_centers_pre[i, 2], data_centers_pre[i,3]] #add 1 to all values
    mu_baseline = data_centers_pre[i,0] 
    print("cluster "+str(i))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    raw_pre_series[label] = kernel_y
    raw_pre_SDs[label] = data_centers_pre[i,4]
    
#rodent mean post
raw_post_series = {}
raw_post_SDs = {}
for i in range(0, len(data_centers_post)):
    label = raw_post_types[i]
    mu_amps = [data_centers_post[i,1], data_centers_post[i, 2], data_centers_post[i,3]] #add 1 to all values
    mu_baseline = data_centers_post[i,0] 
    print("cluster "+str(i))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    raw_post_series[label] = kernel_y
    raw_post_SDs[label] = data_centers_post[i,4]
    
#rodent SVM Pre
svm_pre_series = {}
SVM_pre_SDs = {}
for i in range(0, len(clf_means_pre)):
    label = 'pre '+ unique_pre[i]
    mu_amps = [clf_means_pre[i,1], clf_means_pre[i, 2], clf_means_pre[i,3]] #add 1 to all values
    mu_baseline = clf_means_pre[i,0] 
    print("cluster "+str(i))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    svm_pre_series[label] = kernel_y
    SVM_pre_SDs[label] = clf_means_pre[i,4]
    
#rodent SVM Post
svm_post_series = {}
svm_post_SDs = {}
for i in range(0, len(clf_means_post)):
    label = 'post '+ unique_post[i]
    #filter out types not predicted
    if label not in ['SVM fam84b','SVM rorb','SVM tlx3','SVM vip']:
        mu_amps = [clf_means_post[i,1], clf_means_post[i, 2], clf_means_post[i,3]] #add 1 to all values
        mu_baseline = clf_means_post[i,0] 
        print("cluster "+str(i))
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        svm_post_series[label] = kernel_y
        svm_post_SDs[label] = clf_means_post[i,4]

#--------------------------------------------------
#make one big correlation matrix for post types and optics
#note: because the below uses means there may be nan values in the svm prediction cases when some classes are never predicted
#meaning they don't have a meaningful mean value
all_dicts = [human_series, Rodent_OPTICS_series, raw_pre_series, raw_post_series, svm_pre_series, svm_post_series]
selected_dicts = [human_series, Rodent_OPTICS_series, raw_post_series, svm_post_series]
species_comp_dicts = [Rodent_OPTICS_series, human_series]
species_comp_SD_dicts = [Rodent_OPTICS_SDs, human_SDs]
supervised_unsupervised_comp = [Rodent_OPTICS_series, svm_pre_series, svm_post_series]
supervised_unsupervised_SD_dicts = [Rodent_OPTICS_SDs, SVM_pre_SDs, svm_post_SDs]

species_comp_vals = []
species_comp_SDs = []
species_comp_labels = []

for some_dict in species_comp_dicts:
    for entry in some_dict.keys():
        if not np.isnan(some_dict[entry]).any():
            species_comp_vals.append(some_dict[entry])
            species_comp_labels.append(entry) #store label by row
            
for some_dict in species_comp_SD_dicts:
    for entry in some_dict.keys():
        if not np.isnan(some_dict[entry]).any():
            species_comp_SDs.append(some_dict[entry])

#"""
supervised_unsupervised_comp_vals = []
supervised_unsupervised_comp_SDs = []
supervised_unsupervised_comp_labels = []


for some_dict in supervised_unsupervised_comp:
    for entry in some_dict.keys():
        if not np.isnan(some_dict[entry]).any():
            supervised_unsupervised_comp_vals.append(some_dict[entry])
            supervised_unsupervised_comp_labels.append(entry) #store label by row
            
for some_dict in supervised_unsupervised_SD_dicts:
    for entry in some_dict.keys():
        if not np.isnan(some_dict[entry]).any():
            supervised_unsupervised_comp_SDs.append(some_dict[entry])
#"""
#specificy method comp ordering
method_comp_vals = []
method_comp_SDs = []
method_comp_labels = []

for entry in Rodent_OPTICS_series:
    if not np.isnan(Rodent_OPTICS_series[entry]).any():
        method_comp_SDs.append(Rodent_OPTICS_SDs[entry])
        method_comp_vals.append(Rodent_OPTICS_series[entry])
        method_comp_labels.append(entry) #store label by row

print(svm_post_series)

method_comp_vals.append(svm_pre_series["pre nr5a1"])
method_comp_SDs.append(SVM_pre_SDs["pre nr5a1"])
method_comp_labels.append("pre Nr5a1")

method_comp_vals.append(svm_post_series["post sim1"])
method_comp_SDs.append(svm_post_SDs["post sim1"])
method_comp_labels.append("post Sim1")

method_comp_vals.append(svm_pre_series["pre tlx3"])
method_comp_SDs.append(SVM_pre_SDs["pre tlx3"])
method_comp_labels.append("pre Tlx3")

method_comp_vals.append(svm_post_series["post pvalb"])
method_comp_SDs.append(svm_post_SDs["post pvalb"])
method_comp_labels.append("post Pvalb")

method_comp_vals.append(svm_pre_series["pre ntsr1"])
method_comp_SDs.append(SVM_pre_SDs["pre ntsr1"])
method_comp_labels.append("pre Ntsr1")

method_comp_vals.append(svm_post_series["post nr5a1"])
method_comp_SDs.append(svm_post_SDs["post nr5a1"])
method_comp_labels.append("post Nr5a1")

method_comp_vals.append(svm_post_series["post sst"])
method_comp_SDs.append(svm_post_SDs["post sst"])
method_comp_labels.append("post sst")

"""
method_comp_vals.append(svm_post_series["post pvalb"])
method_comp_SDs.append(svm_post_SDs["post pvalb"])
method_comp_labels.append("post Pvalb")

method_comp_vals.append(svm_post_series["post sst"])
method_comp_SDs.append(svm_post_SDs["post sst"])
method_comp_labels.append("post sst")

method_comp_vals.append(svm_post_series["post vip"])
method_comp_SDs.append(svm_post_SDs["post vip"])
method_comp_labels.append("post vip")

method_comp_vals.append(svm_pre_series["pre sim1"])
method_comp_SDs.append(SVM_pre_SDs["pre sim1"])
method_comp_labels.append("pre sim1")
"""

vals_list = []
labels_list = []
for some_dict in selected_dicts:
    for entry in some_dict.keys():
        if not np.isnan(some_dict[entry]).any():
            vals_list.append(some_dict[entry])
            labels_list.append(entry) #store label by row
      

#plot big matrix
corr_mat_pearson = np.corrcoef(np.asarray(vals_list))


print(corr_mat_pearson)
#plt.matshow(corr_mat_pearson)
fig = plt.figure()


ax = fig.add_subplot(111)
cax = ax.matshow(corr_mat_pearson, interpolation='nearest')
fig.colorbar(cax)

x = range(0, len(labels_list))
x = np.arange(len(labels_list))
#plt.yticks(x, labels_list)
#plt.xticks(x, labels_list)


ax.set_xticks(x)
ax.set_yticks(x)
ax.set_xticklabels(labels_list, rotation = 80)
ax.set_yticklabels(labels_list)

#title = 
save_name = "./fig3/Post_and_OPTICS_Corr_Mat.svg"
plt.savefig(save_name)
#plt.show()


        #----------------------------------------------------------------------

#plot supervised_unsupervised comp
corr_mat_pearson = np.corrcoef(np.asarray(supervised_unsupervised_comp_vals))
similarity_mat = weighted_similarity(corr_mat_pearson, supervised_unsupervised_comp_SDs, rodent_SD_of_SDs, 1, 1)
print(similarity_mat )
#plt.matshow(corr_mat_pearson)
fig = plt.figure()


ax = fig.add_subplot(111)
cax = ax.matshow(similarity_mat , interpolation='nearest')
fig.colorbar(cax)

x = range(0, len(supervised_unsupervised_comp_labels))
x = np.arange(len(supervised_unsupervised_comp_labels))
#plt.yticks(x, labels_list)
#plt.xticks(x, labels_list)


ax.set_xticks(x)
ax.set_yticks(x)
ax.set_xticklabels(supervised_unsupervised_comp_labels, rotation = 80)
ax.set_yticklabels(supervised_unsupervised_comp_labels)

#title = 
fig.set_size_inches((4.42, 3.15))
fig.set_dpi(1200)
fig.tight_layout()
save_name = "./fig4/supervised_unsupervised_Corr_Mat.svg"
plt.savefig(save_name)
#plt.show()

         #----------------------------------------------------------------------

#plot species comp
corr_mat_pearson = np.corrcoef(np.asarray(species_comp_vals))
similarity_mat = weighted_similarity(corr_mat_pearson, species_comp_SDs, joint_SD_of_SDs, 1, 1)
print(similarity_mat)
#plt.matshow(corr_mat_pearson)
#Slice the correlation matrix down to size
similarity_mat = similarity_mat[0:5, 5:]
fig = plt.figure()


ax = fig.add_subplot(111)
cax = ax.matshow(similarity_mat, interpolation='nearest')
fig.colorbar(cax)

#redefine the labels by hand
species_comp_labels[0:5] = ["R0","R1", "R2", "R3", "R4"]

#x = range(0, len(species_comp_labels))
y = np.arange(len(species_comp_labels[0:5]))
x = np.arange(len(species_comp_labels[5:]))
#plt.yticks(x, labels_list)
#plt.xticks(x, labels_list)


ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(species_comp_labels[5:], rotation = 80)
ax.set_yticklabels(species_comp_labels[0:5])

#title = 
fig.set_size_inches((4.0, 3.15))
fig.set_dpi(1200)
fig.tight_layout()
save_name = "./fig3/Human_Rodent_OPTICS_Corr_Mat.svg"
plt.savefig(save_name, transparent=True)
#plt.show()

        #----------------------------------------------------------------------

#plot methods comp
corr_mat_pearson = np.corrcoef(np.asarray(method_comp_vals))
similarity_mat = weighted_similarity(corr_mat_pearson, method_comp_SDs, rodent_SD_of_SDs, 1, 1)
print(similarity_mat)

#Slice the correlation matrix down to size
similarity_mat = similarity_mat[0:5, 5:]

#plt.matshow(corr_mat_pearson)
fig = plt.figure()


ax = fig.add_subplot(111)
cax = ax.matshow(similarity_mat, interpolation='nearest')
fig.colorbar(cax)

#x = range(0, len(method_comp_labels))
#x = np.arange(len(method_comp_labels))
#plt.yticks(x, labels_list)
#plt.xticks(x, labels_list)

x = range(0, len(method_comp_labels[5:]))
y = range(0, len(method_comp_labels[0:5]))

ax.set_xticks(x)
ax.set_yticks(y)
ax.set_xticklabels(method_comp_labels[5:], rotation = 80)
ax.set_yticklabels(method_comp_labels[0:5])

#title = 
fig.set_size_inches((2.797, 3.196))
fig.set_dpi(1200)
fig.tight_layout()
save_name = "./fig4/Method_Comp_Corr_Mat.svg"
plt.savefig(save_name, transparent=True)
#print the matrix to determine slicing
print(corr_mat_pearson)
#plt.show()


#--------------------------------------------------
"""
#time series dictionaries:
all_dicts = [human_series, Rodent_OPTICS_series, raw_pre_series, raw_post_series, svm_pre_series, svm_post_series]
#compute cross correlation matrices for all length 2 combinations but not all permutations
for dict1, dict2 in combinations(all_dicts, 2): 
    #initialize matrix to hold output giving all values -2 to make correlation errors clear
    corr_mat = np.full((len(dict1), len(dict2)), -2)
    np.corrcoef(0)
    dict1_vals = []
    dict1_labels = []
    for entry1 in dict1.keys():
        dict1_vals.append(dict1[entry1])
        dict1_labels.append(entry1)
    
    dict2_vals = []
    dict2_labels = []
    for entry2 in dict2.keys():
        dict2_vals.append(dict2[entry2])
        dict2_labels.append(entry2)
    
    arr_1 = np.asarray(dict1_vals)
    arr_2 = np.asarray(dict2_vals)
    corr_mat_pearson = np.corrcoef(arr_1, arr_2)
    print("arr1:")
    print(arr_1)
    print("arr2")
    print(arr_2)
    print(corr_mat_pearson)
    plt.matshow(corr_mat_pearson)
    y = range(0,len(dict1_labels))
    x = range(0,len(dict2_labels))
    plt.yticks(y, dict1_labels)
    plt.xticks(x, dict2_labels)
    #title = 
    plt.show()
    
    #different (paired, not vectorized) strategy
    index1 = 0
    entry1_list = []
    for entry1 in dict1:
        index2=0
        dict1_vals.append(dict1[entry1])
        entry2_list = []
        for entry2 in dict2:
            print("printing dict1[entry]")
            print(dict1[entry1])
            print("printing dict1[entry].shape")
            print(dict1[entry1].shape)
            print(dict1[entry1].ndim)
            print("printing dict2[entry]")
            print(dict2[entry2])
            print("printing dict2[entry].shape")
            print(dict2[entry2].shape)
            print(dict2[entry2].ndim)
            #correlation = correlate(dict1[entry1], dict2[entry2])
            correlation = np.correlate(dict1[entry1], dict2[entry2])
            print("printing correlation")
            print(correlation)
            corr_mat[index1, index2] = np.nanmean(correlation)
            #corr_mat[index1, index2] = spearmanr(dict1[entry], dict2[entry])
            #corr_mat[index1, index2] = pearsonr(dict1[entry1], dict2[entry2])[0]
            index2 += 1
            entry2_list.append(entry2)
        index1 += 1
        entry1_list.append(entry1)
    print("plotting table "+entry1+" "+entry2)
    print(corr_mat)
    plt.matshow(corr_mat)
    y = range(0,len(entry1_list))
    x = range(0,len(entry2_list))
    plt.yticks(y,entry1_list)
    plt.xticks(x, entry2_list)
    plt.show()
 """   