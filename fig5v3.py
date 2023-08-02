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
import copy

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
from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift 
from sklearn.cluster import OPTICS 
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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
from scipy.signal import lfilter
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

measures_file = open('Measures_ex_1.3mM_Human_Type_Pair_Stim.p', "rb")
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
    pair_id = file_name.split(cut_on)[8] #6 for standard #8 for two phase, 9 for windows
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[6] #4 for standard 6 for two phase, 7 for windows
    pre_type = pre_type.split(cut_on)[0]
    post_type = file_name.split(cut_on)[7] #5 for standard 7 for two phase, 8 for windows
    post_type = post_type.split(cut_on)[0]
    #print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+str(pair_id))
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
    #mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus, mu_scale, sigma_scale, = params
    mu_baseline, mu_amps, mu_taus, SD, mu_scale = params
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
        
       
        if ppr > 4: #1000 for the excessive value, note: this was set to 4 until March 9th when changed for some testing, 4 may still be the best value
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
    new_row.append(SD)
    #new_row = new_row + [mu_tau for mu_tau in mu_taus]
    #new_row.append(sigma_baseline)
    #new_row = new_row +[sigma_amp for sigma_amp in sigma_amps]
    #new_row = new_row + [sigma_tau for sigma_tau in sigma_taus]
    #note: mu_scale=None and therefore not included
    #new_row.append(mu_scale) #just added mu scale for some reason?
    #print("printing mu_scale")
    #print(mu_scale)
    #new_row.append(sigma_scale)
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
model_data = [row[8:] for row in data[:]] #model only
#model_data = [row[8:12] for row in data[:]] #only mu kernel data
#model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "sigma_baseline", "sigma_amp1", "sigma_amp2", "sigma_amp3", "sigma_scale"]
model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "SD", "mu_scale"]
#model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3"]

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

pickle_file = open("Extracted_ex_1.3mM_Human_Type_Pair_Stim.p", "rb")
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
    pca = copy.deepcopy(PCA(whiten=True))
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_arr = scaler.transform(input_data)
    pca_result = pca.fit_transform(scaled_arr)
    
    #km = KMeans(n_clusters=num_clusters, algorithm='full' ).fit(pca_result)
    km = KMeans(n_clusters=num_clusters, algorithm='full' ).fit(scaled_arr)
    #km = AgglomerativeClustering(n_clusters=num_clusters,).fit(pca_result)
    #km = KMeans(n_clusters=3, algorithm='full' ).fit(scaled_arr)
    cluster_centers = km.cluster_centers_
    print("printing cluster centers")
    print(cluster_centers)
    #inverse the PCA to figure out what the centers are
    #centroid_features = pca.inverse_transform(cluster_centers)
    centroid_features = cluster_centers
    print("printing centroid features")
    print(centroid_features)
    labels = km.labels_
    num_by_cluster = [0 for i in range(0, num_clusters)]
    for i in range(0, len(labels)):
        num_by_cluster[labels[i]] = num_by_cluster[labels[i]] +1
    
    return (centroid_features, labels, num_by_cluster, pca, pca_result)    
    """
    cluster_ids = [[] for i in range(0, num_clusters)]
    for row, label in enumerate(labels):
        cluster_ids[label].append(row_labels[row][id_index])
    for i in range(0, len(cluster_ids)):
        print("Cluster "+str(i))
        print(cluster_ids[i])
    """

def surrogate_centroids(input_data, row_labels, cluster_alg, id_index=2):
    pca = PCA(whiten=True)
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_arr = scaler.transform(input_data)
    pca_result = pca.fit_transform(scaled_arr)
    
    clustering = cluster_alg.fit(pca_result)
    labels = clustering.labels_
    #num_clusters = cluster_alg.n_clusters
    num_clusters = len(np.unique(labels))
    cluster_rows = [[] for i in range(0, num_clusters)]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(labels)):
        print("current label:")
        print(labels[i])
        cluster_rows[labels[i]].append(input_data[i][:])

    cluster_medians = [[] for i in range(0, num_clusters)]
    cluster_SDs = [[] for i in range(0, num_clusters)]
    for i in range(0, num_clusters):
        cluster_rows[i] = np.asarray(cluster_rows[i])
        print("printing cluster "+str(i))
        print(cluster_rows[i])
        #the below should return a 1d list of feature means for each cluster 
        cluster_medians[i] = [np.median(cluster_rows[i][:, j]) for j in range(0, len(input_data[0]))]
        cluster_SDs[i] = [np.std(cluster_rows[i][:, j]) for j in range(0, len(input_data[0]))]
    return (np.asarray(cluster_medians),  np.asarray(cluster_SDs), labels)

def data_centers(input_data, row_labels, target_index, id_index=2):
    #num_clusters = cluster_alg.n_clusters
    row_targets = [row[target_index] for row in row_labels]
    unique_types = np.unique(row_targets)
    #index rows with number corresponding to unique type
    row_labels = [-1 for i in range(0, len(row_targets))]
    for i in range(0, len(row_targets)):
        for j in range(0, len(unique_types)):
            if row_targets[i] == unique_types[j]:
                row_labels[i] = j
    num_unique = len(unique_types)
    group_rows = [[] for i in range(0, num_unique)]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(row_labels)):
        #print("current label:")
        #print(row[i])
        group_rows[row_labels[i]].append(input_data[i][:])

    group_means = [[] for i in range(0, num_unique)]
    for i in range(0, num_unique):
        group_rows[i] = np.asarray(group_rows[i])
        #print("printing cluster "+str(i))
        #print(cluster_rows[i])
        #the below should return a 1d list of feature means for each cluster 
        group_means[i] = [np.mean(group_rows[i][:, j]) for j in range(0, len(input_data[0]))]
    return (np.asarray(group_means), row_labels, unique_types, group_rows)
    
#this function generates kernel values over a series of timesteps using the
#forward euler approximation also used in the microstmulation project
def gen_kernel_approx(mu_amps, mu_taus, num_timesteps, mu_baseline=None, dt=1):
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
    return (x_vals, y_vals)
#test cluster centers
print("running cluster centroids")

#mu_amps
num_clusters = 6
model_centroids, cluster_labels, syn_by_cluster, pca, PCA_result  = cluster_centroids(model_data, row_labels, num_clusters=num_clusters)

#set up OPTICS clustering
km = OPTICS(min_samples=10, metric="sqeuclidean") #metric="braycurtis"
#km = AgglomerativeClustering(affinity="l2", linkage="average")
surrogate_means, surrogate_SDs, cluster_labels_surr = surrogate_centroids(model_data, row_labels, km)

#Save params with cluster labels to output csv:
out_table = [[] for i in range(0, len(model_data))]
for i in range(0, len(model_data)):
    out_table[i].append('H'+str(cluster_labels_surr[i]+1)) #cluster
    #out_table[i].append(row_labels[i][0]) #pre type
    #out_table[i].append(row_labels[i][1]) #post type
    for j in range(0, len(model_data[0])):
        out_table[i].append(round(model_data[i][j],3))
    out_table[i].append(cluster_labels_surr[i]+1)
out_arr = np.asarray(out_table)
row_length = len(out_arr[0, :])
ind = np.argsort( out_arr[:,row_length-1] )
out_arr = out_arr[ind] #sort by cluster id
print(out_arr)
np.savetxt("Figures/1.3mMhumanparams.csv", out_arr[:, 0:row_length-1], delimiter=",", fmt='%s')


#acount for OPTICS indexing from -1

num_clusters_surr = len(cluster_labels_surr)
cluster_labels_surr = [cluster_labels_surr[i]+1 for i in range(0, len(cluster_labels_surr))]

#non-optics clustering without offset for outliers
"""
num_clusters_surr = len(cluster_labels_surr)
cluster_labels_surr = [cluster_labels_surr[i] for i in range(0, len(cluster_labels_surr))]
"""

print("Printing surrogate means")
print(surrogate_means)

print("printing surrogate labels")
print(cluster_labels_surr) 

#infer number of synapses for surrogate centroids

runs_by_cluster_surr = [0 for i in range(0, num_clusters_surr)]
syn_by_cluster_surr = [0 for i in range(0, num_clusters_surr)]
for i in range(0, len(row_labels)):
    pair_ID = row_labels[i][2]
    #print(pair_ID)
    cluster_ID = cluster_labels_surr[i]
    syn_by_cluster_surr[cluster_ID] +=1
    #print(cluster_ID)
    runs_by_cluster_surr[cluster_ID] += run_nums_ID[pair_ID]

#-----------------------------------------------------------------------------
#code to get kernel plots of the cre-types in the raw data
"""
data_centers_post, post_ind_types, unique_post_types, group_rows_post = data_centers(model_data, row_labels, 1, id_index=2)
#infer number of synapses for data centers
runs_by_group_post = [0 for i in range(0, len(unique_post_types))]
syn_by_group_post = [0 for i in range(0, len(unique_post_types))]
for i in range(0, len(row_labels)):
    pair_ID = row_labels[i][2]
    #print(pair_ID)
    group_ID = post_ind_types[i]
    syn_by_group_post[group_ID] +=1
    #print(cluster_ID)
    runs_by_group_post[group_ID] += run_nums_ID[pair_ID]
"""

#post type
"""
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values by Post_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers post")
print(data_centers_post)
for i in range(0, len(data_centers_post)):
    if len(group_rows_post[i]) > 4:
        label = ""+unique_post_types[i]+": "+str(syn_by_group_post[i])+" synapses, "+str(runs_by_group_post[i]) +" runs"
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_post[i,1], data_centers_post[i, 2], data_centers_post[i,3]] #add 1 to all values
        mu_baseline = data_centers_post[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label)
        #plt.plot(kernel_x, kernel_y)
plt.legend()
plt.show()

#pre_type
data_centers_pre, pre_ind_types, unique_pre_types, group_rows_pre = data_centers(model_data, row_labels, 0, id_index=2)
#infer number of synapses for data centers
runs_by_group_pre = [0 for i in range(0, len(unique_pre_types))]
syn_by_group_pre = [0 for i in range(0, len(unique_pre_types))]
for i in range(0, len(row_labels)):
    pair_ID = row_labels[i][2]
    #print(pair_ID)
    group_ID = pre_ind_types[i]
    syn_by_group_pre[group_ID] +=1
    #print(cluster_ID)
    runs_by_group_pre[group_ID] += run_nums_ID[pair_ID]


#pre type
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values by pre_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers pre")
print(data_centers_post)
for i in range(0, len(data_centers_pre)):
    if len(group_rows_pre[i]) > 4:
        label = ""+unique_pre_types[i]+": "+str(syn_by_group_pre[i])+" synapses, "+str(runs_by_group_pre[i]) +" runs"
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_pre[i,1], data_centers_pre[i, 2], data_centers_pre[i,3]] #add 1 to all values
        mu_baseline = data_centers_pre[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label)
        #plt.plot(kernel_x, kernel_y)
plt.legend()
plt.show()
"""


#-----------------------------------------------------------------------------


#model_centroids, cluster_labels, syn_by_cluster, PCA, PCA_result  = cluster_centroids(phys_data, row_labels, num_clusters=num_clusters) 
#build supervised learner
#set up training and target data
train_set = PCA_result
train_target_post = [row[1] for row in row_labels] 
train_target_pre = [row[0] for row in row_labels] 
print("printing traing_target_pre")
print(train_target_pre)
unique_post = np.unique(train_target_post)
unique_pre = np.unique(train_target_pre)
#unique_post = np.unique(train_target_pre) #change for post (also terrible naming hack)
print("printing unique_post")
print(unique_post)
num_unique_post = len(unique_post)
num_unique_pre = len(unique_pre)

def supervised_means(train_set, train_target, unique_post, num_unique_post):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    clf=SVC()
    clf.fit(train_set, train_target) #change for post
    clf_predictions = clf.predict(train_set)
    prediction_rows = [[] for i in range(0, len(train_set[0,:]))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(clf_predictions)):
        print("current label:")
        print(clf_predictions[i])
        for j in range(0, len(unique_post)):
            if clf_predictions[i] == unique_post[j]:
                prediction_rows[j].append(model_data[i][:])
    
    prediction_means = [[0 for j in range(0, len(model_data[0]))] for i in range(0, num_unique_post)]
    for i in range(0, num_unique_post):
        if len(prediction_rows[i]) > 1:
            prediction_rows[i] = np.asarray(prediction_rows[i])
            print("printing cluster "+str(i))
            print("cluster type: "+str(unique_post[i]))
            print(prediction_rows[i])
            #the below should return a 1d list of feature means for each cluster 
            prediction_means[i] = [np.mean(prediction_rows[i][:, j]) for j in range(0, len(model_data[0]))]
    print("printing cluster_means")
    print(prediction_means)
    return (np.asarray(prediction_rows), np.asarray(prediction_means))

"""
prediction_rows_pre, clf_means_pre = supervised_means(train_set, train_target_pre, unique_pre, num_unique_pre)
prediction_rows_post, clf_means_post = supervised_means(train_set, train_target_post, unique_post, num_unique_post)
"""

#----------------------------------------------------------------

def supervised_means_scatter(train_set, train_target, unique_post, num_unique_post):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    clf=SVC()
    clf.fit(train_set, train_target) #change for post
    clf_predictions = clf.predict(train_set)
    prediction_rows = [[] for i in range(0, len(train_set[0,:]))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(clf_predictions)):
        print("current label:")
        print(clf_predictions[i])
        for j in range(0, len(unique_post)):
            if clf_predictions[i] == unique_post[j]:
                prediction_rows[j].append(train_set[i][:])
    
    prediction_means = [[0 for j in range(0, len(train_set[0]))] for i in range(0, num_unique_post)]
    for i in range(0, num_unique_post):
        if len(prediction_rows[i]) > 1:
            prediction_rows[i] = np.asarray(prediction_rows[i])
            print("printing cluster "+str(i))
            print("cluster type: "+str(unique_post[i]))
            print(prediction_rows[i])
            #the below should return a 1d list of feature means for each cluster 
            prediction_means[i] = [np.mean(prediction_rows[i][:, j]) for j in range(0, len(train_set[0]))]
    print("printing cluster_means")
    print(prediction_means)
    return (np.asarray(prediction_rows), np.asarray(prediction_means))
"""
def unsupervised_scatter(train_set):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    cluster_alg = OPTICS(min_samples=5, metric="sqeuclidean")
    cluster_alg.fit(train_set) 
    cluster_predictions = cluster_alg.labels_
    prediction_rows = [[] for i in range(0, len(np.unique(cluster_predictions)))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(cluster_predictions)):
        prediction_rows[cluster_predictions[i]].append(train_set[i][:])
    for i in range(0, len(prediction_rows)):
        if len(prediction_rows[i]) > 1:
            prediction_rows[i] = np.asarray(prediction_rows[i])
    return np.asarray(prediction_rows)
"""
def unsupervised_scatter(train_set):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    cluster_alg = OPTICS(min_samples=10, metric="sqeuclidean")
    cluster_alg.fit(train_set) 
    cluster_predictions = cluster_alg.labels_
    prediction_rows = [[] for i in range(0, len(np.unique(cluster_predictions)))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(cluster_predictions)):
        prediction_rows[cluster_predictions[i]].append(train_set[i][:])
    for i in range(0, len(prediction_rows)):
        print("prediction "+str(i)+" length = "+str(len(prediction_rows[i])))
        if len(prediction_rows[i]) > 1:
            prediction_rows[i] = np.asarray(prediction_rows[i])
            print(prediction_rows[i])
        else:
            print("conversion failed for prediction "+str(i))
    #return np.asarray(prediction_rows)
    return prediction_rows


"""
prediction_rows_pre, clf_means_pre = supervised_means(train_set, train_target_pre, unique_pre, num_unique_pre)
prediction_rows_post, clf_means_post = supervised_means(train_set, train_target_post, unique_post, num_unique_post)
"""

"""
#clf1 = LogisticRegression(max_iter=10000)
clf1 = SVC()
clf1.fit(train_set, train_target_pre)
pre_params = clf1.get_params()
inverse_coef1 = pca.inverse_transform(clf1.coef_)

print("printing pre coef_ PCA inverse")
print(inverse_coef1)
print("printing intercepts")
print(clf1.intercept_)
print("pre class order")
print(clf1.classes_)
print("feature order")
#print(model_labels )
#print(phys_labels)
#out_data_pre = np.hstack((inverse_coef1, np.transpose(clf1.intercept_)))
out_data_pre = np.c_[inverse_coef1, np.transpose(clf1.intercept_)]
print("stacked np data")
print(out_data_pre)
#col_labels = phys_labels
#col_labels = model_labels[0:4]
col_labels = model_labels
col_labels.append("intercept")
print("col_labels")
print(col_labels)
#print(phys_labels.append("intercept"))
pd_out_pre = pd.DataFrame(out_data_pre, columns=col_labels)
pd_out_pre = pd_out_pre.set_index(clf1.classes_)
pd_out_pre.to_csv('./LR_fit_coefficients/Model_Mu_Pre_Class.csv')
print(pd_out_pre)



#clf2 = LogisticRegression(max_iter=10000)
clf2 = SVC()
clf2.fit(train_set, train_target_post)
post_params = clf2.get_params()
inverse_coef2 = pca.inverse_transform(clf2.coef_)
print("printing post coef_ PCA inverse")
print(inverse_coef2)
print("printing intercepts")
print(clf2.intercept_)
print("post class order")
print(clf2.classes_)
print("feature order")
#print(model_labels)
print(phys_labels)


out_data_post = np.c_[inverse_coef2, np.transpose(clf2.intercept_)]
print("stacked np data")
print(out_data_post)
#print(phys_labels.append("intercept"))
pd_out_post = pd.DataFrame(out_data_post, columns=col_labels)
pd_out_post = pd_out_post.set_index(clf2.classes_)
pd_out_post.to_csv('./LR_fit_coefficients/Model_Mu_Post_Class.csv')
print("printing col_labels")
print(col_labels)
print(pd_out_post)
"""

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

#infer number of synapses for centroids
"""
runs_by_cluster = [0 for i in range(0, num_clusters)]
for i in range(0, len(row_labels)):
    pair_ID = row_labels[i][2]
    #print(pair_ID)
    cluster_ID = cluster_labels[i]
    #print(cluster_ID)
    runs_by_cluster[cluster_ID] += run_nums_ID[pair_ID]
"""

#check the reference cluster:
"""
for i in range(0, num_clusters):
    if syn_by_cluster[i] == 4:
        print("reference cluster: "+str(i))
        for j in range(0, len(cluster_labels)):
            if cluster_labels[j] == i:
                print("pre_type = "+row_labels[j][0]+" post_type = "+row_labels[j][1] + " ID = "+str(row_labels[j][2]))
"""

#taus (time constants for the kernels)
mu_kernel_taus = [15, 200, 300]  #orginally [15, 200, 650] for both mu an sigma
sigma_kernel_taus = [15, 100, 300]

"""
plt.figure()
plt.title("Kernel of SRP centroid with baseline")
x = [i for i in range(-30,1000)]
#kernels_over_time = []
for i in range(0, len(model_centroids)):
    label = "cluster "+str(i)+": "+str(syn_by_cluster[i])+" synapses, "+str(runs_by_cluster[i]) +" runs"
    mu_amps = [model_centroids[i,1], model_centroids[i,2], model_centroids[i,3]] #add 1 to all values
    mu_baseline = model_centroids[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    plt.plot(kernel_x, kernel_y, label=label)
plt.legend()
plt.show()
"""  
  
"""
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
"""

#supervised plots
"""
#pre type
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values of SVM Predictions Pre_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_pre)):
    if len(prediction_rows_pre[i]) > 1:
        print("selected: "+str(unique_pre[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [clf_means_pre[i,1], clf_means_pre[i, 2], clf_means_pre[i,3]] #add 1 to all values
        mu_baseline = clf_means_pre[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=unique_pre[i])
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_pre[i])
plt.legend()
plt.show()

#post type
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values of SVM Predictions Post_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_post)):
    if len(prediction_rows_post[i]) > 1:
        print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [clf_means_post[i,1], clf_means_post[i, 2], clf_means_post[i,3]] #add 1 to all values
        mu_baseline = clf_means_post[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=unique_post[i])
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_post[i])
plt.legend()
plt.show()
"""

#save values for later plotting:
    
#unsupervised
filename = 'saved_kernels/OPTICSHuman_means_post_1.3mM.pickle'
outfile = open(filename,'wb')
pickle.dump(surrogate_means, outfile)
outfile.close()

filename = 'saved_kernels/OPTICSHuman_SDs_post_1.3mM.pickle'
outfile = open(filename,'wb')
pickle.dump(surrogate_SDs, outfile)
outfile.close()

filename = 'saved_kernels/OPTICSHuman_syn_by_cluster_surr_1.3mM.pickle'
outfile = open(filename,'wb')
pickle.dump(syn_by_cluster_surr, outfile)
outfile.close()

filename = 'saved_kernels/OPTICSHuman_runs_by_cluster_surr_1.3mM.pickle'
outfile = open(filename,'wb')
pickle.dump(runs_by_cluster_surr, outfile)
outfile.close()

#save model data
filename = 'saved_kernels/Human_model_data.pickle'
outfile = open(filename,'wb')
pickle.dump(model_data, outfile)
outfile.close()



plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernels by Cluster Human")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(surrogate_means)):
    label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
    #label = "cluster "+str(i)
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    plt.plot(kernel_x, kernel_y, label=label)
    #plt.plot(kernel_x, kernel_y)
#plt.legend()
plt.show()  


#---------    
cluster_predictions = unsupervised_scatter(train_set) 

f = plt.figure()
#plt.title("clusters, model PCA")
#plt.title("Human Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
# disabling xticks by Setting xticks to an empty list
plt.xticks([]) 
 
# disabling yticks by setting yticks to an empty list
plt.yticks([]) 
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=label)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
#plt.xlabel("PC1")
#plt.ylabel("PC2")
save_name = "./Figures/fig5/Human_scatter.svg"
f.set_size_inches((5.5, 4.3))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)
#plt.legend()
plt.show()
#---------

cluster_predictions = unsupervised_scatter(train_set) 

f = plt.figure()
#plt.title("clusters, model PCA")
plt.title("Human Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(cluster_predictions[i][:, 2], cluster_predictions[i][:, 3], label=label)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
plt.xlabel("PC3")
plt.ylabel("PC4")
save_name = "./Figures/fig5/Human_scatter_pc34.svg"
f.set_size_inches((3.76, 3.15))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)
#plt.legend()
plt.show()

#----------------------------------------------------------------
cluster_predictions = unsupervised_scatter(train_set) 

f = plt.figure()
#plt.title("clusters, model PCA")
plt.title("Human Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(cluster_predictions[i][:, 1], cluster_predictions[i][:, 2], label=label)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
plt.xlabel("PC2")
plt.ylabel("PC3")
save_name = "./Figures/fig5/Human_scatter_pc23.svg"
f.set_size_inches((3.76, 3.15))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)
#plt.legend()
plt.show()

#----------------------------------------------------------------
cluster_predictions = unsupervised_scatter(train_set) 

f = plt.figure()
#plt.title("clusters, model PCA")
plt.title("Human Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(cluster_predictions[i][:, 1], cluster_predictions[i][:, 3], label=label)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
plt.xlabel("PC2")
plt.ylabel("PC4")
save_name = "./Figures/fig5/Human_scatter_pc24.svg"
f.set_size_inches((3.76, 3.15))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)
#plt.legend()
plt.show()

#----------------------------------------------------------------
#scatter plots by predicted type
prediction_rows_pre_sct, clf_means_pre_sct = supervised_means_scatter(train_set, train_target_pre, unique_pre, num_unique_pre)
prediction_rows_post_sct, clf_means_post_sct = supervised_means_scatter(train_set, train_target_post, unique_post, num_unique_post)


#scatter plots by predicted type
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Model Predicted class Pre_type, PCA")
ax = plt.subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_pre_sct)):
    if isinstance(prediction_rows_pre_sct[i], np.ndarray):
        #print("selected: "+str(unique_pre[i]))
        #print("printing prediction_rows")
        #rint(prediction_rows_pre)
        ax.scatter(prediction_rows_pre_sct[i][:, 0], prediction_rows_pre_sct[i][:, 1], label=unique_pre[i])
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_pre[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
save_name = "./Figures/fig5/Model_cluster_pre_scatter.svg"
f.set_dpi(600)
f.set_size_inches((2.56, 3.35))
#f.t1ght_layout()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(save_name)
#plt.legend()
plt.show()


#----------------------------------------------------------------

f = plt.figure()
plt.title("Model Predicted class Post_type, PCA")
ax = plt.subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_pre_sct)):
    if isinstance(prediction_rows_post_sct[i], np.ndarray):
        #print("selected: "+str(unique_pre[i]))
        #print("printing prediction_rows")
        #print(prediction_rows_pre)
        ax.scatter(prediction_rows_post_sct[i][:, 0], prediction_rows_post_sct[i][:, 1], label=unique_post[i])
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_pre[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
save_name = "./Figures/fig5/Model_cluster_post_scatter.svg"
f.set_dpi(600)
f.set_size_inches((2.56, 3.35))
#f.t1ght_layout()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(save_name)
#plt.legend()
plt.show()


pca = PCA(whiten=True)
scaler = StandardScaler()
scaler.fit(model_data)
scaled_arr = scaler.transform(model_data)
pca_result = pca.fit_transform(scaled_arr)
cluster_predictions = unsupervised_scatter(pca_result)

plt.figure()
plt.title("Rodent Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(cluster_predictions)):
    if isinstance(prediction_rows_post_sct[i], np.ndarray):
        plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=str(i))
        #plt.plot(kernel_x, kernel_y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
#----------------------------------------------------------------

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

