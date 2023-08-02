#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:10:08 2021

@author: john
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import pickle
import copy
import scipy.stats as stat
import math
import sys
import itertools
import random
import math

from srplasticity.srp import (
    ExpSRP,
    ExponentialKernel,
    _convolve_spiketrain_with_kernel,
    get_stimvec,
)
from srplasticity.inference import fit_srp_model_gridsearch
#loaded tm to deal with conditional below, clean this up
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from itertools import cycle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyClassifier
from scipy.stats import sem
from scipy.stats import wilcoxon
from scipy.stats import kruskal
from scipy.stats import normaltest
from scipy.stats import f_oneway
from sklearn.utils import shuffle


#these come from example code and may not be needed
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


#some code taken from towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python
#----------------------------------------------------------------------------------

row_info = []
data = []
pyr_types = ['nr5a1', 'rorb']
pyr_indices = []

l5et_types = ['sim1', 'fam84b']
l5et_indices = []

data_post_pre = {}
unique_pre = set()
unique_post = set()
unique_pairs = set()

measures_file = open('Measures_ex_1.3mM_Mouse_Type_Pair_Stim.p', "rb")
#measures_file = open('Measures_ex_2mM_Mouse_Type_Pair_Stim.p', "rb")
#print('printing measures dict:')
measures_dict = pickle.load(measures_file)
#print(measures_dict)
num_100s = 0
for i in range(1, len(sys.argv)):
    cut_on = '_'
    file_name = sys.argv[i]
    #print(i)
    #print(file_name)
    pair_id = file_name.split(cut_on)[6] #6 for standard
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[4] #4 for standard
    pre_type = pre_type.split(cut_on)[0]
    post_type = file_name.split(cut_on)[5] #5 for standard
    post_type = post_type.split(cut_on)[0]
    #print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+str(pair_id))
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus, mu_scale, sigma_scale, = params
    new_row = []
    #add identifiers
    new_row.append(pre_type) #1
    new_row.append(post_type) #2
    new_row.append(pair_id) #3
    
    unique_pre.add(pre_type)
    unique_post.add(post_type)
    unique_pairs.add(pre_type+post_type)
    
    #add measures: phys 2 version
    """
    try:
        print('attempted key = ')
        print((pre_type, post_type))
        print("keys = ")
        print(measures_dict[(pre_type, post_type)][pair_id].keys())
        
        key_50hz = ('ic', 50.0, 0.25)

        #select only trials with 100hz data for comparison
        key_100hz = ('ic', 100.0, 0.25)
        key_50hz = ('ic', 50.0, 0.25)
        if key_100hz in measures_dict[(pre_type, post_type)][pair_id].keys():
            num_100s += 1
            print("100hz entry: "+str(num_100s))
        else:
            print("no 100")
            continue
        
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        print("printing ppr:" +str(ppr))
        #skip the excessively high ppr
        
        if ppr > 4: #1000 for the excessive value
            print("skipper ppr = "+str(ppr))
            continue
       
        areas = measures_dict[(pre_type, post_type)][pair_id]['areas_50hz_mean']
        print("printing area:" +str(areas))
        #this first fifth is having trouble, values are clearly wrong (way too high)
        #replacement of min and max values needs to be fixed in extraction code
        release_prob = measures_dict[(pre_type, post_type)][pair_id]['release_prob_all']
        print("printing releaase_prob:" +str(release_prob))
        #print("printing first_fifth:" +str(first_fifth))
        first_fifth = measures_dict[(pre_type, post_type)][pair_id]['first_fifth_50hz_mean']
        first_second = measures_dict[(pre_type, post_type)][pair_id]['first_second_50hz_mean']
        #new_row.append(ppr) #4
        new_row.append(areas) #5
        new_row.append(release_prob) #6
        #new_row.append(first_fifth) #7
        new_row.append(first_second) #8
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id][key_50hz ]['recovery'] 
        new_row.append(recovery_50) #9
        

        #100hz measures

        areas_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['area_first_eight_mean'] #10
        ppr_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['first_second_mean'] #11
        first_fifth_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['first_fifth_mean'] #12
        recovery_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['recovery'] #13
        new_row.append(areas_100) #10
        new_row.append(ppr_100) #11
        #new_row.append(first_fifth_100) #12
        new_row.append(recovery_100) #13

        
    except:
        print("triggered exception")
        #adjust this so it does not add all missing rows if only some are missing
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
    """
    
    #add measures
    try:
        key_50hz = ('ic', 50.0, 0.25)
        #print('attempted key = ')
        #print((pre_type, post_type))
        #print("keys = ")
        #print(measures_dict[(pre_type, post_type)][pair_id].keys())
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        #print("printing ppr:" +str(ppr))
        #skip the excessively high ppr
        
        if ppr > 4: #1000 for the excessive value
            print("skipper ppr = "+str(ppr))
            continue
       
        areas = measures_dict[(pre_type, post_type)][pair_id]['areas_50hz_mean']
        #print("printing area:" +str(areas))
        #this first fifth is having trouble, values are clearly wrong (way too high)
        #replacement of min and max values needs to be fixed in extraction code
        release_prob = measures_dict[(pre_type, post_type)][pair_id]['release_prob_all']
        #print("printing releaase_prob:" +str(release_prob))
        #print("printing first_fifth:" +str(first_fifth))
        first_fifth = measures_dict[(pre_type, post_type)][pair_id]['first_fifth_50hz_mean']
        first_second = measures_dict[(pre_type, post_type)][pair_id]['first_second_50hz_mean']
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id][key_50hz ]['recovery']
        #new_row.append(ppr)
        new_row.append(areas)
        new_row.append(release_prob)
        new_row.append(first_fifth)
        new_row.append(first_second)
        new_row.append(recovery_50)

    except:
        #adjust this so it does not add all missing rows if only some are missing
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
    
    
    #add model params
    new_row.append(mu_baseline)
    new_row = new_row + [mu_amp for mu_amp in mu_amps]
    #new_row = new_row + [mu_tau for mu_tau in mu_taus]
    new_row.append(sigma_baseline)
    new_row = new_row +[sigma_amp for sigma_amp in sigma_amps]
    #new_row = new_row + [sigma_tau for sigma_tau in sigma_taus]
    #note: mu_scale=None and therefore not included
    #new_row.append(mu_scale) #just added mu scale for some reason?
    #print("printing mu_scale")
    #print(mu_scale)
    new_row.append(sigma_scale)
    data.append(new_row)
    #print(params)[row[1] for row in cell_label_sorted[i]]
    #print("Printing new_row")
    #print(new_row)
    if post_type in data_post_pre:
        if pre_type in data_post_pre[post_type]:
            curr_data = data_post_pre[post_type][pre_type]
            data_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(new_row)))
        else:
             data_post_pre[post_type][pre_type] = np.array(new_row)
    else:
        data_post_pre[post_type] = {pre_type: np.array(new_row)}

#print(data)
num_cols = len(data[0])
#print(data[:,3:])
#params_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
#params_data = [row[3:] for row in data[:]] #model and phys
params_data = [row[8:] for row in data[:]] #model only

phys_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
hybrid_data = [row[3:] for row in data[:]] #model and phys
#model_data = [row[8:] for row in data[:]] #model only
model_data = [row[9:12] for row in data[:]] #only mu kernel data
model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "sigma_baseline", "sigma_amp1", "sigma_amp2", "sigma_amp3", "sigma_scale", "mu_scale" ]

phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

print("printing model data[0]")
print(model_data[0])
#params_data = [row[3:8] for row in data[:]]
#params_data = [row[3:10] for row in data[:]]


#params_data = [row[8:] for row in data[:]]
#row_labels = [row[0:8] for row in data[:]]
row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)



#--------------------------------------------------------------------------
#create dicts of additional synapse information
#produces: target_dict, training_stim_dict

pre_type = 'nr5a1'
post_type = 'nr5a1'

pickle_file = open("Extracted_ex_Mouse_Type_Pair_Stim.p", "rb")
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
recordings = pickle.load(pickle_file)
#print(recordings)
#unneeded_pre_types = ['nr5a1', 'unknown'] 
target_dict = {}
training_stim_dict = {} 
run_nums_ID = {}
for type_pair in recordings.keys():      
    type1, type2 = type_pair
    """
    if type1 != pre_type:
        continue
    if type2 != post_type:
        continue  
    """
    chosen_dict = recordings[type_pair] #should be sst
    #print(chosen_dict)
        
    #target_arr_20 = np.asarray(target_list_20)
    #target_arr_100 = np.asarray(target_list_100)
    #trim these to first 8
    #print(chosen_dict[('ic', 20.0, 0.253)])
    #target_dict = {}
    #training_stim_dict = {} #put this back into use on lines 289 and 290 for version 2
    #print(chosen_dict.keys())
    #print(target_arr_20)
    #stuff below is from version 2
    
    #print("About to print keys")
    #print(chosen_dict.keys())
    for pair_id in chosen_dict.keys():
        #print(key)
        #print(chosen_dict['pair_IDs'])
        num_runs = 0
        if pair_id != 'pair_IDs':
            """
            try: 
                target_dict[pair_id]
            except:
                target_dict[pair_id] = { "pair_ID": pair_id}
            """
            #rint(key)
            #print(chosen_dict[key])
            #clamp, freq, delay = key #unpack tuple key for conditions
            #clamp, freq, delay = key
            #if clamp == 'ic': #select only current clamp
                #try:
                #    #np.append(target_dict[str(int(freq))], chosen_dict[key][:, 0:8])
                #determine row wise average for divisor
                
                #modify to work by pairs
            first_spike_list = []
            testing_counter = 0
            for protocol in chosen_dict[pair_id]:
                """
                try: 
                    target_dict[pair_id]
                except:
                    target_dict[pair_id] = { "pair_ID": pair_id}
                """
                testing_counter += 1
                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol
                    #print(protocol)
                    #first_spike_list = []
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            #changed to make average excluding rows with any column values below 1E-9
                            """
                            safe_row = True 
                            for n in range(0, 8):
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                            """
                            divisor = chosen_dict[pair_id][protocol][i, 0]
                            #print("divisor = "+str(divisor))
                            #if divisor > 1E-6:
                            if divisor > 1E-9:
                                first_spike_list.append(divisor)
                            else: 
                                first_spike_list.append(1E-9)
                        #if len(first_spike_list) > 0:
            #print(testing_counter)
            #print(pair_id)
            #print(protocol)
            #print(len(first_spike_list))
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list)/len(first_spike_list)
                #apply normalisation
            else:
                print("voltage clamp runs only for pair")
            for protocol in chosen_dict[pair_id]:
                #for i in range(0, len(first_spike_list)):
                #divisor = chosen_dict[key][i, 0]
                #print("divisor = "+str(divisor))
                if not isinstance(protocol, int):
                    #print(protocol)
                    clamp, freq, delay = protocol
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i,:]
                            for n in range(0, len(added_row)):
                                """
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                                """
                                if chosen_dict[pair_id][protocol][i, n] < 1E-9:
                                    added_row[n] = 1E-9
                            #if divisor > 1E-5:
                            normed_row = added_row/averaged_divisor
                            #print(normed_row)
                            num_runs += len(added_row)
                            try:
                                target_dict[pair_id][(freq, delay)] = np.vstack((target_dict[pair_id][(freq, delay)], normed_row))
                                #print("try")
                                #print(target_dict)
                                #print("append success")
                            except:
                                #target_dict[str(int(freq))] = chosen_dict[key][:, 0:8]
                                #print(freq)
                                try:
                                    target_dict[pair_id][(freq, delay)] = normed_row
                                except: 
                                    target_dict[pair_id] = {(freq, delay): normed_row}
                                #print(int(freq))
                                try:
                                    training_stim_dict[pair_id][(freq, delay)] = [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3 #should be [0] + ...
                                except:
                                    training_stim_dict[pair_id] = {(freq, delay): [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3}
        run_nums_ID[pair_id] = num_runs

#----------------------------------------------------------------------------------




# define func to partition list into even subsets
# source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def partition_list(lst, num_partitions):
        partitions_list  = [lst[i:i + num_partitions] for i in range(0, len(lst), num_partitions)]
        return partitions_list

def pval_to_str(pval):
    if pval >= 0.05:
        return "p = "+str(round(pval, 3))
    elif ((pval < 0.05) and (pval >= 0.01)):
        return "p < 0.05"
    elif ((pval < 0.01) and (pval >=0.001)):
        return "p<0.01"
    else:
        return "p<0.001"


#test different cluster theories
def list_cluster_pairs(input_data, row_labels, num_clusters=3, id_index=2):
    pca = PCA(whiten=True)
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_arr = scaler.transform(input_data)
    pca_result = pca.fit_transform(scaled_arr)
    
    km = KMeans(n_clusters=num_clusters, algorithm='full' ).fit(pca_result)
    #km = KMeans(n_clusters=3, algorithm='full' ).fit(scaled_arr)
    #cluster_centers_indices = km.cluster_centers_indices_
    labels = km.labels_
    cluster_ids = [[] for i in range(0, num_clusters)]
    for row, label in enumerate(labels):
        cluster_ids[label].append(row_labels[row][id_index])
    for i in range(0, len(cluster_ids)):
        print("Cluster "+str(i))
        print(cluster_ids[i])

#get results on cluster centroids
def cluster_centroids(input_data, row_labels, num_clusters=3, id_index=2):
    pca = PCA(whiten=True)
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_arr = scaler.transform(input_data)
    pca_result = pca.fit_transform(scaled_arr)
    
    km = KMeans(n_clusters=num_clusters, algorithm='full' ).fit(pca_result)
    #km = AgglomerativeClustering(n_clusters=num_clusters,).fit(pca_result)
    #km = KMeans(n_clusters=3, algorithm='full' ).fit(scaled_arr)
    cluster_centers = km.cluster_centers_
    print("printing cluster centers")
    print(cluster_centers)
    #inverse the PCA to figure out what the centers are
    centroid_features = pca.inverse_transform(cluster_centers)
    print("printing centroid features")
    print(centroid_features)
    labels = km.labels_
    num_by_cluster = [0 for i in range(0, num_clusters)]
    for i in range(0, len(labels)):
        num_by_cluster[labels[i]] = num_by_cluster[labels[i]] +1
    
    return (centroid_features, labels, num_by_cluster)    
    """
    cluster_ids = [[] for i in range(0, num_clusters)]
    for row, label in enumerate(labels):
        cluster_ids[label].append(row_labels[row][id_index])
    for i in range(0, len(cluster_ids)):
        print("Cluster "+str(i))
        print(cluster_ids[i])
    """

def gen_kernel(mu_amps, mu_taus, num_timesteps, mu_baseline=None, dt=1):
    if mu_baseline == None:
        kernels = [0 for i in range(0,30)]
        for step in range(0, num_timesteps):
            kernel = 0
            for i in range(0, len(mu_amps)):
                kernel = kernel + mu_amps[i]*math.exp((-1*step*dt)/mu_taus[i])
            kernels.append(kernel)
        return kernels
    else:
        kernels = [mu_baseline for i in range(0,30)]
        for step in range(0, num_timesteps):
            kernel = 0
            for i in range(0, len(mu_amps)):
                kernel = kernel + mu_amps[i]*math.exp((-1*step*dt)/mu_taus[i])
            kernel = kernel +mu_baseline
            kernels.append(kernel)
        return kernels

#test cluster centers
print("running cluster centroids")

#mu_amps
num_clusters = 5
model_centroids, cluster_labels, syn_by_cluster = cluster_centroids(model_data, row_labels, num_clusters=num_clusters)
#print(cluster_labels)
#make kernel from kernel params
#mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
#sigma_kernel_taus = [15, 100, 300]

"""
params_by_cluster = [[] for i in range(0, num_clusters)]
mean_params_by_cluster = [[0 for j in range(0, 10)] for i in range(0, num_clusters)]
for i in range(0, num_clusters):
    for j in range(0, len(cluster_labels)):
        if cluster_labels[j] == i:
            params_by_cluster[i].append(model_data[j][:])
    params_by_cluster[i] = np.vstack(params_by_cluster[i])
    for k in range(0, len(model_data[0,:])):
        #should yield 1x10 list of mean model parameters
        mean_params_by_cluster[i][k] = np.mean(params_by_cluster[i][k]) 
"""

runs_by_cluster = [0 for i in range(0, num_clusters)]
for i in range(0, len(row_labels)):
    pair_ID = row_labels[i][2]
    #print(pair_ID)
    cluster_ID = cluster_labels[i]
    #print(cluster_ID)
    runs_by_cluster[cluster_ID] += run_nums_ID[pair_ID]


#check the reference cluster:
for i in range(0, num_clusters):
    if syn_by_cluster[i] == 4:
        print("reference cluster: "+str(i))
        for j in range(0, len(cluster_labels)):
            if cluster_labels[j] == i:
                print("pre_type = "+row_labels[j][0]+" post_type = "+row_labels[j][1] + " ID = "+str(row_labels[j][2]))


#taus (time constants for the kernels)
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
sigma_kernel_taus = [15, 100, 300]

plt.figure()
plt.title("Kernel of SRP centroid with baseline")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(model_centroids)):
    label = "cluster "+str(i)+": "+str(syn_by_cluster[i])+" synapses, "+str(runs_by_cluster[i]) +" runs"
    mu_amps = [model_centroids[i,0], model_centroids[i,1], model_centroids[i,2]] #add 1 to all values
    mu_baseline = model_centroids[i,0] 
    kernel = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    kernels_over_time.append(kernel)
    plt.plot(x, kernel, label=label)
plt.legend()
plt.show()
    
plt.figure()
plt.title("Kernel of SRP centroid without baseline")
for i in range(0, len(model_centroids)):
    label = "cluster "+str(i)+" "+str(syn_by_cluster[i])+" synapses"
    mu_amps = [model_centroids[i,1], model_centroids[i,2], model_centroids[i,3]]
    mu_baseline = model_centroids[i,0] 
    kernel = gen_kernel(mu_amps, mu_kernel_taus, 1000, dt=1)
    kernels_over_time.append(kernel)
    plt.plot(x, kernel, label=label)
plt.legend()
plt.show()

    
    
fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(projection='3d')

#color_map = plt.get_cmap('gist_rainbow')
#print(color_map)
#ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
marker = [',', '+', '.', 'o', '*', 'v', '8', '4', '|' ]
#labels
title_name = "Cluster centroids by mu amp" 
x_label = model_labels[1],
y_label = model_labels[2],
z_label = model_labels[3],
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
x= model_centroids[:,1]
y= model_centroids[:,2]
z= model_centroids[:,3]
ax.set_title(title_name)
for i in range(0, len(model_centroids)):
    ax.scatter(x[i], y[i], z[i], marker=marker[i])
plt.show()

#sig amps
#model_centroids = cluster_centroids(model_data, row_labels, num_clusters=5)

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(projection='3d')

#color_map = plt.get_cmap('gist_rainbow')
#print(color_map)
#ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
marker = [',', '+', '.', 'o', '*', 'v', '8', '4', '|' ]
#labels
title_name = "Cluster centroids by sigma amp" 
x_label = model_labels[4],
y_label = model_labels[5],
z_label = model_labels[6],
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
x= model_centroids[:,4]
y= model_centroids[:,5]
z= model_centroids[:,6]
ax.set_title(title_name)
for i in range(0, len(model_centroids)):
    ax.scatter(x[i], y[i], z[i], marker=marker[i])
plt.show()

#mu_baseline, sig_baseline, sig_scale
#model_centroids = cluster_centroids(model_data, row_labels, num_clusters=5)

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(projection='3d')

#color_map = plt.get_cmap('gist_rainbow')
#print(color_map)
#ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
marker = [',', '+', '.', 'o', '*', 'v', '8', '4', '|' ]
#labels
title_name = "Cluster centroids by mu_baseline, sigma_baseline, sigma_scale" 
x_label = model_labels[0],
y_label = model_labels[4],
z_label = model_labels[8],
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
x= model_centroids[:,0]
y= model_centroids[:,4]
z= model_centroids[:,8]
ax.set_title(title_name)
for i in range(0, len(model_centroids)):
    ax.scatter(x[i], y[i], z[i], marker=marker[i])
plt.show()


"""
#phys plots

#area, stp ind, ppr
phys_centroids = cluster_centroids(phys_data, row_labels, num_clusters=5)

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(projection='3d')

#color_map = plt.get_cmap('gist_rainbow')
#print(color_map)
#ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
marker = [',', '+', '.', 'o', '*', 'v', '8', '4', '|' ]
#labels
title_name = "Area, STP Induction, and PPR" 
x_label = phys_labels[0],
y_label = phys_labels[2],
z_label = phys_labels[3],
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
x= phys_centroids[:,0]
y= phys_centroids[:,2]
z= phys_centroids[:,3]
ax.set_title(title_name)
for i in range(0, len(phys_centroids)):
    ax.scatter(x[i], y[i], z[i], marker=marker[i])
plt.show()

#areas, rlease_prob, rec 50

fig = plt.figure()
fig.set_size_inches(12, 12)
ax = fig.add_subplot(projection='3d')

#color_map = plt.get_cmap('gist_rainbow')
#print(color_map)
#ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
marker = [',', '+', '.', 'o', '*', 'v', '8', '4', '|' ]
#labels
title_name = "Cluster centroids by areas, release_probability, and 50Hz Recovery" 
x_label = phys_labels[0],
y_label = phys_labels[1],
z_label = phys_labels[4],
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
x= phys_centroids[:,0]
y= phys_centroids[:,1]
z= phys_centroids[:,4]
ax.set_title(title_name)
for i in range(0, len(phys_centroids)):
    ax.scatter(x[i], y[i], z[i], marker=marker[i])
plt.show()
"""


"""
print("phys case (special) 2 clusters")
list_cluster_pairs(phys_data, row_labels, num_clusters=2)

print("model, 5 clusters")
list_cluster_pairs(model_data, row_labels, num_clusters=5)

print("hybrid case, 2 clusters")
list_cluster_pairs(hybrid_data, row_labels, num_clusters=2)
"""

