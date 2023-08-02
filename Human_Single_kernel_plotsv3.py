# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:28:42 2022

@author: jgben
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:10:08 2021

@author: john
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
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
    pair_id = file_name.split(cut_on)[8] #6 for standard, 8 two step, 9 on windows
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[6] #4 for standard, 6 two step, 7 on windows
    pre_type = pre_type.split(cut_on)[0]
    post_type = file_name.split(cut_on)[7] #5 for standard, 7 two step, 8 on windows
    post_type = post_type.split(cut_on)[0]
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
        #skip the excessively high pprmodel_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "SD", "mu_scale"]
        
       
        if ppr > 4: #1000 for the excessive value, note: this was set to 4 until March 9th when changFdataed for some testing, 4 may still be the best value
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

mu_kernel_taus = [15, 200, 300]

#define colours for plotting
red='xkcd:blood red' #e50000
orange='xkcd:pumpkin'
green='xkcd:apple green'
blue='xkcd:cobalt'
purple='xkcd:barney purple'
brown = 'xkcd:brown'
pink='xkcd:hot pink'
taupe='xkcd:taupe'

#colour_set = [red, orange, green, blue, purple, brown]
#create different colour set from rodents
#colour_set = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',]
#pair similar colours to rodent
colour_set = ['tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'xkcd:sky blue', 'tab:blue' ,'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue',]
#colour_set = mpl.colormaps['dark2']

single_kernel_dir = "./Figures/single_kernels/"

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
    for i in range(0, num_clusters):
        cluster_rows[i] = np.asarray(cluster_rows[i])
        print("printing cluster "+str(i))
        print(cluster_rows[i])
        #the below should return a 1d list of feature means for each cluster 
        cluster_medians[i] = [np.median(cluster_rows[i][:, j]) for j in range(0, len(input_data[0]))]
    return (np.asarray(cluster_medians), labels, pca_result)

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
    #set pre-spike baseline, figure out why this isn't always the case
    for i in range(0, 2000):
        y_vals[i] = mu_baseline
    return (x_vals, y_vals)

def save_single_kernel(mu_amps, mu_baseline, title_name, file_name, colour="xkcd:blue"):
    f = plt.figure()
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.title("Kernel of SRP OPTICS mean values with baseline") "Kernels by Cluster Human"
    plt.title(title_name)
    
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #plt.plot(kernel_x, kernel_y, label=label, color=colour_set[i]) #match colour to bar plots
    plt.plot(kernel_x, kernel_y, color=colour) #match colour to bar plots
    #plt.plot(kernel_x, kernel_y, label=label) #no colour match
    
    #ax.set_ylim([-6.5, 30])
    f.set_size_inches((1.4, 1.4))
    f.set_dpi(1200)
    f.tight_layout()
    plt.savefig(file_name, transparent=True)
    
#------------------------------------------------------------------------------

#set up OPTICS clustering
km = OPTICS(min_samples=10, metric="sqeuclidean") #metric="braycurtis" num=6
#km = OPTICS(min_samples=8, metric="braycurtis") #metric="braycurtis"
#km = AgglomerativeClustering(affinity="l2", linkage="average")
surrogate_means, cluster_labels_surr, train_set = surrogate_centroids(model_data, row_labels, km)    

#create single 2D list with all cluster labels and data
#merged_arr = labels.copy() #labels, row_labels, plot_arr
merged_arr = [[label] for label in cluster_labels_surr]
for i in range(0, len(merged_arr )):
    merged_arr[i] = merged_arr[i] + row_labels[i]
    merged_arr[i] = merged_arr[i] + [entry for entry in model_data[i]]
num_clusters = len(set(cluster_labels_surr))
pre_index = 1
post_index = 2

#------------------------------------------------------------------------------

#plot human scatter using clustering above
prediction_rows = [[] for i in range(0, len(np.unique(cluster_labels_surr)))]
#create list of lists corresponding to all synapse fit rows by cluster

for i in range(0, len(cluster_labels_surr)):
    prediction_rows[cluster_labels_surr[i]].append(train_set[i][:])
for i in range(0, len(prediction_rows)):
    print("prediction "+str(i)+" length = "+str(len(prediction_rows[i])))
    if len(prediction_rows[i]) > 1:
        prediction_rows[i] = np.asarray(prediction_rows[i])
        print(prediction_rows[i])
    else:
        print("conversion failed for prediction "+str(i))

cluster_predictions = prediction_rows
f = plt.figure()
#plt.title("clusters, model PCA")
plt.title("Human Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=label, color=colour_set[i])
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
#hide ticks
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
    
#plt.xlabel("PC1")
#plt.ylabel("PC2")
plt.legend()
save_name = "./Figures/fig5/Human_scatter_matched.svg"
f.set_size_inches((3.76, 3.15))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)

#------------------------------------------------------------------------------

#plot overlay for verification
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
    plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=label)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
plt.xlabel("PC1")
plt.ylabel("PC2")
#save_name = "./Figures/fig5/Human_scatter.svg"
f.set_size_inches((3.76, 3.15))
f.set_dpi(1200)
f.tight_layout()
#plt.savefig(save_name)
#plt.legend()
plt.show()

#------------------------------------------------------------------------------

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
    #runs_by_cluster_surr[cluster_ID] += run_nums_ID[pair_ID]

#-----------------------------------------------------------------------------
#plot overlay kernels for verification
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
    plt.plot(kernel_x, kernel_y, label=label, color=colour_set[i])
    #plt.plot(kernel_x, kernel_y)
#plt.legend()
plt.show()

#-----------------------------------------------------------------------------

#plot single kernels
for i in range(0, len(surrogate_means)):
    #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    file_name = single_kernel_dir + "human_cluster_"+str(i)+".svg"
    title_name = "H"+str(i)
    print("len(surrogate_means) = "+str(len(surrogate_means)))
    save_single_kernel(mu_amps, mu_baseline, title_name, file_name, colour=colour_set[i])