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
from scipy.stats import mode
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
    print(file_name)
    pair_id = file_name.split(cut_on)[5] 
    pair_id = int(pair_id.split(".p")[0])
    print("pair_id: "+str(pair_id))
    pre_type = file_name.split(cut_on)[3] 
    pre_type = pre_type.split(cut_on)[0]
    pre_type = pre_type.split("/")[1]
    print("pre_type: "+str(pre_type))
    post_type = file_name.split(cut_on)[4] 
    post_type = post_type.split(cut_on)[0]
    print("post_type: "+str(post_type))
    
    #remove gabaergic pre_types since we only want excitatory pairs
    if pre_type == 'sst':
        print("skipping sst pre")
        continue
    if pre_type == 'pvalb':
        print("skipping pvab pre")
        continue
    if pre_type == 'vip':
        print("skipping vip pre")
        continue
    
    #print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+str(pair_id))
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
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
model_data = [row[8:13] for row in data[:]] #model only
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

pickle_file = open("Extracted_STP_1.3mM_Rodent.p", "rb")
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


#mapping to match rodent clusters

#April19th version
transcriptomic_colours = {
    'pvalb':'seagreen',
    #'nr5a1':'orangered',
    'nr5a1':'firebrick',
    #'sim1':'firebrick',
    'sim1':'orangered',
    #'sst':"sandybrown",
    'sst':"darkorchid",
    'vip':"silver",
    'ntsr1':'deepskyblue',
    'tlx3':'forestgreen',
    'fam84b':"lightpink",
    'rorb':'xkcd:dark teal'
    }

"""
transcriptomic_colours = {
    'pvalb':'seagreen',
    'nr5a1':'darkorchid',
    'sim1':'firebrick',
    'sst':"sandybrown",
    'vip':"silver",
    'ntsr1':'deepskyblue',
    'tlx3':'forestgreen',
    'fam84b':"lightpink",
    'rorb':'xkcd:dark teal'
    }
"""

#map cre-types to capital case
cre_capital = {
    'pvalb':'Pvalb',
    'nr5a1':'Nr5a1',
    'sim1':'Sim1',
    'sst':'Sst',
    'vip':'Vip',
    'ntsr1':'Ntsr1',
    'tlx3':'Tlx3',
    'fam84b':'Fam84b',
    'rorb':'Rorb'
    }
#----------------------------------------------------------------------------------
#define functions
def data_centers(input_data, row_labels, target_index, id_index=2):
    #num_clusters = cluster_alg.n_clusters
    row_targets = [row[target_index] for row in row_labels]
    unique_types = np.unique(row_targets)
    #index rows with number corresponding to unique type
    new_row_labels = [-1 for i in range(0, len(row_targets))]
    for i in range(0, len(row_targets)):
        for j in range(0, len(unique_types)):
            if row_targets[i] == unique_types[j]:
                new_row_labels[i] = j
    num_unique = len(unique_types)
    group_rows = [[] for i in range(0, num_unique)]
    group_ids = [[] for i in range(0, num_unique)]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(new_row_labels)):
        #print("current label:")
        #print(row[i])
        group_rows[new_row_labels[i]].append(input_data[i][:])
        group_ids[new_row_labels[i]].append(row_labels[i][id_index])

    group_means = [[] for i in range(0, num_unique)]
    group_modes = [[] for i in range(0, num_unique)]
    for i in range(0, num_unique):
        group_rows[i] = np.asarray(group_rows[i])
        #print("printing cluster "+str(i))
        #print(cluster_rows[i])
        #the below should return a 1d list of feature means for each cluster 
        group_means[i] = [np.mean(group_rows[i][:, j]) for j in range(0, len(input_data[0]))]
        
        #alternate for multi dim binned mode
        print(group_rows[i][:, 0:4]) 
        H, edges = np.histogramdd(group_rows[i][:, 0:4], bins= (3,3,3,3))
        print(H.shape)
        print(edges)
        #find bin with most entries
        max_ind = np.unravel_index(H.argmax(), H.shape)
        print(max_ind)
        mode_values = [0 for j in range(0, len(max_ind))]
        for j in range(0, len(max_ind)):
            #interpolate value
            dim_ind = max_ind[j]
            mode_values[j] = (edges[j][dim_ind] + edges[j][dim_ind+1])/2
        group_modes[i] = mode_values
        
    return (np.asarray(group_modes), new_row_labels, unique_types, group_rows,group_ids)
    #return (np.asarray(group_means), new_row_labels, unique_types, group_rows,group_ids)


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

#----------------------------------------------------------------------------------
#pre-type plots


#id_index determines cre_type of synapse selected (pre or post type)
data_centers_pre, pre_ind_types, unique_pre_types, group_rows_pre, group_ids = data_centers(model_data, row_labels, 0, id_index=2)

#code for kernel plot
#pre type
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
"""
plt.figure()
plt.title("Kernel of SRP values by Post_type")
"""
"""
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers post")
print(data_centers_pre)
for i in range(0, len(data_centers_pre)):
    #i determines the selected post_type
    if len(group_rows_pre[i]) > 3: #select only groups with at least 4 samples
        plt.figure()
        ax = plt.subplot(111)
        plt.title("Pre type Kernel of SRP values for "+str(unique_pre_types[i])+" "+str(len(group_rows_pre[i]))+" samples")
        #plot individual kernels
        individual_pres = group_rows_pre[i]
        #print("printing weird synapses")
        for j in range(0, len(group_rows_pre[i])):
            #label = "mean val"
            #if len(prediction_rows_post[i]) > 4:
            #print("selected: "+str(unique_post[i]))
            #print(prediction_rows[i])
            #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
            mu_amps = [individual_pres[j,1], individual_pres[j, 2], individual_pres[j,3]] #add 1 to all values
            mu_baseline = individual_pres[j,0] 
            kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
            #kernels_over_time.append(kernel)
            plt.plot(kernel_x, kernel_y, color='gray')
            #plt.plot(kernel_x, kernel_y)
            print("mu amps: "+str(individual_pres[j,1])+" " +str(individual_pres[j, 2]) +" " +str(individual_pres[j,3]))
            print("baseline: "+str(individual_pres[j,0] ))
            
            #if mu_baseline < -4:
            #    print(group_ids[i][j])
            
            
        #plot mean kernel
        label = "mean val"
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_pre[i,1], data_centers_pre[i, 2], data_centers_pre[i,3]] #add 1 to all values
        mu_baseline = data_centers_pre[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label, color="black")
        #plt.plot(kernel_x, kernel_y)
        
        #else:
        #    print(unique_post[i])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend()
        
        #uncomment to plot
        #plt.show()
"""     
#----------------------------------------------------------------
        
#plots PC space
#remove this section to run without scaling + pca
pca = PCA(whiten=True)

#scale data before pca
scaler = StandardScaler()
scaler.fit(model_data)
scaled_arr = scaler.transform(model_data)
pca_arr = pca.fit_transform(scaled_arr)
pre_data = pca_arr.tolist()
data_centers_pre, pre_ind_types, unique_pre_types, group_rows_pre, group_ids = data_centers(pre_data, row_labels, 0, id_index=2)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax = plt.subplot(111)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

plt.title("Pre")
for i in range(0, len(data_centers_pre)):
    #i determines the selected post_type
    #if len(group_rows_pre[i]) > 3: #select only groups with at least 4 samples
    #plot individual kernels
    individual_pres = group_rows_pre[i]
    #print("printing weird synapses")
    ax1 = individual_pres[:, 0] #PC1
    ax2 = individual_pres[:, 1] #PC1
    label = str(unique_pre_types[i])
    ax.scatter(ax1, ax2, label=cre_capital[label], color=transcriptomic_colours[label])
plt.xlabel("R-PC1")
plt.ylabel("R-PC2")
f.set_dpi(1200)
f.set_size_inches((3.30, 2.40))
f.tight_layout()
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig2/model_pre_scatter.svg"
plt.savefig(save_name, transparent=True)
#plt.show()
        
#----------------------------------------------------------------

#plots PC space
#remove this section to run without scaling + pca
pca = PCA(whiten=True)

#scale data before pca
scaler = StandardScaler()
scaler.fit(phys_data)
scaled_arr = scaler.transform(phys_data)
pca_arr = pca.fit_transform(scaled_arr)
pre_data = pca_arr.tolist()
data_centers_pre, pre_ind_types, unique_pre_types, group_rows_pre, group_ids = data_centers(pre_data, row_labels, 0, id_index=2)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax = plt.subplot(111)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

plt.title("Pre-type Phys")
for i in range(0, len(data_centers_pre)):
    #i determines the selected post_type
    #if len(group_rows_pre[i]) > 3: #select only groups with at least 4 samples
    #plot individual kernels
    individual_pres = group_rows_pre[i]
    #print("printing weird synapses")
    ax1 = individual_pres[:, 0] #PC1
    ax2 = individual_pres[:, 1] #PC1
    label = str(unique_pre_types[i])
    ax.scatter(ax1, ax2, label=label)
plt.xlabel("PC1")
plt.ylabel("PC2")
f.set_dpi(600)
f.set_size_inches((2.9, 2))
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#f.tight_layout()
#plt.legend()
save_name = "./fig2/phys_pre_scatter.svg"
plt.savefig(save_name)
#plt.show()
        
            
#------------------------------------------------------------------------------

#post type plots
        
data_centers_post, post_ind_types, unique_post_types, group_rows_post, group_ids = data_centers(model_data, row_labels, 1, id_index=2)
#plot for pre_types      
"""
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers post")
print(data_centers_post)
for i in range(0, len(data_centers_post)):
    #i determines the selected post_type
    if len(group_rows_post[i]) > 3: #select only groups with at least 4 samples
        plt.figure()
        ax = plt.subplot(111)
        plt.title("Post type Kernel of SRP values for "+str(unique_post_types[i])+" "+str(len(group_rows_post[i]))+" samples")
        #plot individual kernels
        individual_posts = group_rows_post[i]
        print("printing weird synapses")
        for j in range(0, len(group_rows_post[i])):
            #label = "mean val"
            #if len(prediction_rows_post[i]) > 4:
            #print("selected: "+str(unique_post[i]))
            #print(prediction_rows[i])
            #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
            mu_amps = [individual_posts[j,1], individual_posts[j, 2], individual_posts[j,3]] #add 1 to all values
            mu_baseline = individual_posts[j,0] 
            kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
            #kernels_over_time.append(kernel)
            plt.plot(kernel_x, kernel_y, color='gray')
            #plt.plot(kernel_x, kernel_y)
            print("mu amps: "+str(individual_posts[j,1])+" " +str(individual_posts[j, 2]) +" " +str(individual_posts[j,3]))
            print("baseline: "+str(individual_posts[j,0] ))
            
            #if mu_baseline < -4:
            #    print(group_ids[i][j])
            
            
        #plot mean kernel
        label = "mean val"
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_post[i,1], data_centers_post[i, 2], data_centers_post[i,3]] #add 1 to all values
        mu_baseline = data_centers_post[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label, color="black")
        #plt.plot(kernel_x, kernel_y)
        
        #else:
        #    print(unique_post[i])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend()
        
        #uncomment to show
        #plt.show()
"""

#----------------------------------------------------------------
        
#plots PC space
#remove this section to run without scaling + pca
pca = PCA(whiten=True)

#plot for model params
#scale data before pca
scaler = StandardScaler()
scaler.fit(model_data)
scaled_arr = scaler.transform(model_data)
pca_arr = pca.fit_transform(scaled_arr)
post_data = pca_arr.tolist()
data_centers_post, post_ind_types, unique_posttypes, group_rows_post, group_ids = data_centers(post_data, row_labels, 1, id_index=2)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

#ax = plt.subplot(111)
#plt.title("Post-type SRP")
plt.title("Post")
for i in range(0, len(data_centers_post)):
    #i determines the selected post_type
    #if len(group_rows_pre[i]) > 3: #select only groups with at least 4 samples
    #plot individual kernels
    individual_posts = group_rows_post[i]
    #print("printing weird synapses")
    ax1 = individual_posts[:, 0] #PC1
    ax2 = individual_posts[:, 1] #PC1
    label = str(unique_post_types[i])
    #print(label)
    #print(len(individual_posts))
    #print(len(ax1))
    ax.scatter(ax1, ax2, label=cre_capital[label], color=transcriptomic_colours[label])
plt.xlabel("R-PC1")
plt.ylabel("R-PC2")
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend()
#ax.legend(bbox_to_anchor=(1, 0.5))
#plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
#f.set_size_inches((3.11, 2.49))
#f.set_dpi(600)
f.set_dpi(1200)
f.set_size_inches((3.30, 2.40))
f.tight_layout()
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig2/Model_post_scatter.svg"
plt.savefig(save_name, transparent=True)
#plt.show()
        
#---------------------------------------------------------------

#plot for phys params
#scale data before pca
scaler = StandardScaler()
scaler.fit(phys_data)
scaled_arr = scaler.transform(phys_data)
pca_arr = pca.fit_transform(scaled_arr)
post_data = pca_arr.tolist()
data_centers_post, post_ind_types, unique_posttypes, group_rows_post, group_ids = data_centers(post_data, row_labels, 1, id_index=2)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.75, box.height*0.9])

"""
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
"""
#ax = plt.subplot(111)
plt.title("Post-Type Phys")
for i in range(0, len(data_centers_post)):
    #i determines the selected post_type
    #if len(group_rows_pre[i]) > 3: #select only groups with at least 4 samples
    #plot individual kernels
    individual_posts = group_rows_post[i]
    #print("printing weird synapses")
    ax1 = individual_posts[:, 0] #PC1
    ax2 = individual_posts[:, 1] #PC1
    label = str(unique_post_types[i])
    #print(label)
    #print(len(individual_posts))
    #print(len(ax1))
    ax.scatter(ax1, ax2, label=label)
    #plt.scatter(ax1, ax2, label=label)
plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.legend()
#ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), prop={'size': 1})
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
"""
f.set_size_inches((3.11, 2.49))
f.set_dpi(600)
"""
f.set_dpi(600)
f.set_size_inches((2.9, 2))
#f.tight_layout()
#plt.legend()
save_name = "./fig2/Phys_post_scatter.svg"
plt.savefig(save_name)
#plt.show()


