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
#import umap
import random

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
from sklearn import metrics
from itertools import cycle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from scipy.stats import sem
from sklearn.cluster import OPTICS

#these come from example code and may not be needed
import time
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier as rf
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
for i in range(1, len(sys.argv)):
    cut_on = '_'
    file_name = sys.argv[i]
    print(file_name)
    pair_id = file_name.split(cut_on)[9] #6 for standard 8 for two step
    pair_id = pair_id.split(".p")[0]
    pre_type = file_name.split(cut_on)[7] #4 for standard 6 for two step
    pre_type = pre_type.split(cut_on)[0]
    post_type = file_name.split(cut_on)[8] #5 for standard 7 for two step
    post_type = post_type.split(cut_on)[0]
    #print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+pair_id)
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
    mu_baseline, mu_amps, mu_taus, sigma_baseline, sigma_amps, sigma_taus, mu_scale, sigma_scale, = params
    new_row = []
    new_row.append(pre_type)
    new_row.append(post_type)
    
    unique_pre.add(pre_type)
    unique_post.add(post_type)
    unique_pairs.add(pre_type+post_type)
    
    new_row.append(pair_id)
    new_row.append(mu_baseline)
    new_row = new_row + [mu_amp for mu_amp in mu_amps]
    #new_row = new_row + [mu_tau for mu_tau in mu_taus]
    new_row.append(sigma_baseline)
    new_row = new_row +[sigma_amp for sigma_amp in sigma_amps]
    #new_row = new_row + [sigma_tau for sigma_tau in sigma_taus]
    #note: mu_scale=None and therefore not included
    new_row.append(sigma_scale)
    data.append(new_row)
    #print(params)
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
params_data = [row[3:6] for row in data[:]]
row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)



#--------------------------------------------------------------------------
pca = PCA(whiten=True)

#scale data before pca
scaler = StandardScaler()
scaler.fit(params_arr)
scaled_arr = scaler.transform(params_arr)
pca_result = pca.fit_transform(scaled_arr)

#pca_result = pca.fit_transform(params_arr)

#print PCA metrics
print("PCA components: ")
print(pca.components_)
print("PCA explained Variance: ")
print(pca.explained_variance_)
print("Explained Variance ratios")
print(pca.explained_variance_ratio_)
#tsne_results = pca_result

#--------------------------------------------------------------------------
#define colours for plotting
red='xkcd:blood red' #e50000
orange='xkcd:pumpkin'
green='xkcd:apple green'
blue='xkcd:cobalt'
purple='xkcd:barney purple'
brown = 'xkcd:brown'

colour_set = [red, orange, green, blue, purple, brown]

#--------------------------------------------------------------------------

#organiza PCA into dict form
unique_posts = []
pca_post_pre = {}
for i in range(0, len(params_data)):
    pre_type = row_labels[i][0]
    post_type = row_labels[i][1]
    if not post_type in unique_posts:
        unique_posts.append(post_type)
    row = pca_result[i]
    if post_type in pca_post_pre:
        if pre_type in pca_post_pre[post_type]:
            curr_data = pca_post_pre[post_type][pre_type]
            pca_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(row)))
        else:
             pca_post_pre[post_type][pre_type] = np.array(row)
    else:
        pca_post_pre[post_type] = {pre_type: np.array(row)}

raw_post_pre = {}    
for i in range(0, len(params_data)):
    pre_type = row_labels[i][0]
    post_type = row_labels[i][1]
    row = params_data[i]
    if post_type in raw_post_pre:
        if pre_type in raw_post_pre[post_type]:
            curr_data = raw_post_pre[post_type][pre_type]
            raw_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(row)))
        else:
             raw_post_pre[post_type][pre_type] = np.array(row)
    else:
        raw_post_pre[post_type] = {pre_type: np.array(row)}

#plot 
L4pyr_types = ['nr5a1', 'rorb']

l5et_types = ['sim1', 'fam84b']


#fig, ax = plt.subplots()

"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

color_map = plt.get_cmap('gist_rainbow')
#NUM_COLORS = 15
NUM_COLORS = 4
#plotting_data['num_colors'] = NUM_COLORS 
#ax.set_color_cycle([color_map(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1,NUM_COLORS)))
colours = plt.cm.brg(np.linspace(0,1,NUM_COLORS))

#ax.set_prop_cycle(marker=['o', '+', '2', 'x', 'v', '8', 'P', 'D', '4', '|' ])
#marker = itertools.cycle((',', '+', '.', 'o', '*', 'v', '8', '4', '|' )) 
markers = ['4', '+', '.', 'o', '*', 'v', '8', '|']


#create dictionary of unique markers    
marker_dict = {}
which_marker = 0
for label in unique_posts:
    marker_dict[label] = markers[which_marker]
    which_marker += 1

"""
def fac_dep_labels(arr):
    for i in range(0, len(arr)):
        x = [i for i in range(0,8)]
        y = arr[i, 0:8]
        reg = LinearRegression().fit(x, y)
        print(reg.get_params())
        #incomplete: get slope and if postive facilitatin, if negative, depressing
        #determine some criteria for neither, consider comparing w/ allen code

"""
#plot function
def add_to_plot_3D(ax, label, marker, dim1, dim2, dim3):
    ax.scatter( 
        dim1,
        dim2,
        dim3,
        label=label,
        marker = next(marker))
"""

"""
#plot function
def add_to_plot_3D(ax, label, colour, post_labels, dim1, dim2, dim3):
    try:
        #print("in first")
        for i in range(0, len(dim1)):
            #print(post_labels[i])
            marker = marker_dict[post_labels[i]]
            #print(marker)
            ax.scatter( 
                dim1[i],
                dim2[i],
                dim3[i],
                #label=label,
                color = colour,
                marker = marker)
    except:
        #print("in second")
        marker = marker_dict[post_labels]
        ax.scatter( 
                dim1,
                dim2,
                dim3,
                #label=label,
                color = colour,
                marker = marker)
    
desired_dims = [0,1,2]
"""

#
def working_plot1(desired_dims, vals_dict, num_colours, axis_labels, title):
    fig = plt.figure()
    fig.set_size_inches(12, 12)
    ax = fig.add_subplot(projection='3d')
    color_map = plt.get_cmap('gist_rainbow')
    #print(color_map)
    ax.set_prop_cycle('color',plt.cm.brg(np.linspace(0,1, num_colours)))
    marker = itertools.cycle((',', '+', '.', 'o', '*', 'v', '8', '4', '|' ))
    #labels
    title_name = title
    x_label = axis_labels[0],
    y_label = axis_labels[1],
    z_label = axis_labels[2],
    pre_post = False
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.title(title_name)
    for post_type in vals_dict:
        """
        if post_type == 'pvalb':
            for pre_type in vals_dict[post_type]:
                print("pre = "+pre_type+" post= "+post_type)
                label = pre_type+"_"+post_type
                try:
                    dim1 = vals_dict[post_type][pre_type][:, desired_dims[0]]
                    dim2 = vals_dict[post_type][pre_type][:, desired_dims[1]]
                    dim3 = vals_dict[post_type][pre_type][:, desired_dims[2]]
                    add_to_plot_3D(ax, label, marker, dim1, dim2, dim3)
                except:
                    dim1 = vals_dict[post_type][pre_type][desired_dims[0]]
                    dim2 = vals_dict[post_type][pre_type][desired_dims[1]]
                    dim3 = vals_dict[post_type][pre_type][desired_dims[2]]
                    add_to_plot_3D(ax, label, marker, dim1, dim2, dim3)
        else:
        """
        if True:
            #print(post_type)
            aggregated_post_list = [vals_dict[post_type][entry] for entry in vals_dict[post_type]]
            plot_data = np.vstack(aggregated_post_list)
            dim1 = plot_data [:, desired_dims[0]]
            dim2 = plot_data [:, desired_dims[1]]
            dim3 = plot_data [:, desired_dims[2]]
            label = post_type
            add_to_plot_3D(ax, label, marker, dim1, dim2, dim3)
    ax.legend()
    plt.show()

#run clustering via affinity propagation
#code from scikit-learn examples
# Compute Affinity Propagation
plot_arr = pca_result

#--------------------------------------------------------------------

#Apply clustering
"""
af = AffinityPropagation().fit(plot_arr)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
"""

#setp 2021 version used kmeans clustering
"""
km = KMeans(n_clusters=3, random_state=0).fit(plot_arr)
#cluster_centers_indices = km.cluster_centers_indices_
labels = km.labels_
"""

km = OPTICS(min_samples=8, metric="sqeuclidean").fit(plot_arr)
labels = km.labels_



#--------------------------------------------------------------------

#sort data by cluster

label_sorted = {}
cell_label_sorted = {}
#plot_arr = pca_result
#print("len(labels) = "+str(len(labels)))
#print(labels)
for i in range(0, len(labels)):
    if labels[i] in label_sorted:
        label_sorted[labels[i]] = np.vstack((label_sorted[labels[i]], plot_arr[i]))
        #print("if")
        #print(label_sorted[labels[i]])
        #print(post_label_sorted[labels[i]])
        cell_label_sorted[labels[i]].append(row_labels[i])
        #print(cell_label_sorted)
    else:
        #print("else")
        label_sorted[labels[i]] = plot_arr[i]
        cell_label_sorted[labels[i]] = [row_labels[i]]
        #print([row_labels[i][:]])

#--------------------------------------------------------------------
#this section deals with cross_val stability measures

# define func to partition list into even subsets
# source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def partition_list(lst, num_partitions):
        partitions_list  = [lst[i:i + num_partitions] for i in range(0, len(lst), num_partitions)]
        return partitions_list

#cross val for cluster stability measures: trained accuracy by split

#0 data struct to store pair cluster responses, do I need these pairs at first?
#1 random split process
#2 cross val loop with splits for accuracy measure, track also which pairs go where
num_partitions = 10
num_row_labels = 4
cluster_test_numbers = 15
cluster_num_accuracies = []
cluster_num_sems = []
pre_proc_condition = "PCA, Model Parameters"

#function from: https://stackoverflow.com/questions/19385639/duplicate-items-in-legend-in-matplotlib
def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def test_cre_clusters(labeled_arr, cre_types, cre_entry_index, num_clusters, cre_title):
    """
    labeled_arr: array with cluster labels as per merged_arr below
    
    cre_types: set of unique cre_types to check
    
    cre_entry_index: index in arr of cre_type to test against
    """
    cluster_label_index = 0
    cre_dist = {}
    cre_types = list(cre_types)
    print("printing cre_types")
    print(cre_types)
    for cre_type in cre_types:
        cre_dist[cre_type] = [ 0 for i in range(0, num_clusters)] #initialize with 0 entry for each cluster
    for cre_type in cre_types:
        for i in range(0, len(labeled_arr)):
            cluster_label = labeled_arr[i][cluster_label_index]
            cre_label = labeled_arr[i][cre_entry_index]
            print("cre_type is "+cre_label)
            if cre_label == cre_type:
                cre_dist[cre_label][cluster_label] = cre_dist[cre_label][cluster_label] + 1 #increment counter for cre_label at cluster
    
    #sort cre_dist to order by magnitude
    """
    for cre_type in cre_dist:
        cre_dist[cre_type].sort(reverse=True)
    """
    
    width = 1/(num_clusters+1)  # the width of the bars

    fig, ax = plt.subplots()
    for j in range(0, len(cre_types)):
        for i in range(0, num_clusters):
            value = cre_dist[cre_types[j]][i]
            rects1 = ax.bar(j -(width*num_clusters/2) + (i+0.5)*width, value, width, label=str(i), color=colour_set[i])
    
    x = np.arange(len(cre_types)) 
    ax.set_xticks(x)
    ax.set_xticklabels(cre_types)
    
    #plot 
    #plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
    title  = "Number by cluster of cre "+cre_title + pre_proc_condition + " for "+str(num_clusters)+" clusters" 
    plt.title(title)
    plt.ylabel("Num pairs")
    legend_without_duplicate_labels(ax)
    plt.show()
    
    
#create single 2D list with all cluster labels and data
#merged_arr = labels.copy() #labels, row_labels, plot_arr
merged_arr = [[label] for label in labels]
for i in range(0, len(merged_arr )):
    merged_arr[i] = merged_arr[i] + row_labels[i]
    merged_arr[i] = merged_arr[i] + [entry for entry in plot_arr[i]]
num_clusters = len(set(labels))
pre_index = 1
post_index = 2
test_cre_clusters(merged_arr, unique_pre, pre_index, num_clusters, "pre type")
test_cre_clusters(merged_arr, unique_post, post_index, num_clusters, "post type")
"""
for j in range(2, cluster_test_numbers+1):
    km_test = KMeans(n_clusters=j, random_state=0).fit(plot_arr)
    #cluster_centers_indices = km.cluster_centers_indices_
    num_test_labels = km_test.labels_
    print("printing num_test_labels")
    print(num_test_labels)
    
    crossval_accuracies = []
    crossval_metrics = {}
    
    #create single 2D list with all cluster labels and data
    #merged_arr = labels.copy() #labels, row_labels, plot_arr
    merged_arr = [[label] for label in num_test_labels]
    for i in range(0, len(merged_arr )):
        merged_arr[i] = merged_arr[i] + row_labels[i]
        merged_arr[i] = merged_arr[i] + [entry for entry in plot_arr[i]]
    
    #generate plots for stability by cluster num
    #cre_types: unique_pre or unique_post
    cre_entry_index = 2 #1 for pre, 2 for post
    #cre_title = "pre_type "
    cre_title = "post_type "
    test_cre_clusters(merged_arr, unique_pre, cre_entry_index, j, cre_title)
    
    
    
    #repeat the operations below for ten random subset generation repetitions
    for i in range(0, 10):
    
        #divide data into ten random subsets without replacement
        arr_shuffled = merged_arr.copy()
        random.shuffle(arr_shuffled)
    
        
        partitioned_list  = partition_list(arr_shuffled, num_partitions)
        for n in range(0, num_partitions):
            test_set = partitioned_list[n]
            #get all subsets for training set
            train_subsets = partitioned_list[:n]+partitioned_list[n+1:] #is this valid? What about edge cases?
            #append all training set subsets to create single 2D list
            train_set = [e for sbl in train_subsets for e in sbl] #also sanity check this
            #both sets now assumed to be 2D lists
            
            #train random forest on train set
            clf = LogisticRegression()
            #clf = rf(max_depth=2, random_state=0)
            print("printing train_set[0][0]")
            print(train_set[0][0])
            print("printing list comp for training set")
            print([row[0] for row in train_set])
            
            train_labels = [row[0] for row in train_set]
            test_labels = [row[0] for row in test_set]
            train_data = [row[num_row_labels+1:] for row in train_set]
            test_data = [row[num_row_labels+1:] for row in test_set]
            
            #print("printing test data")
            print(test_data)
            
            clf.fit(train_data, train_labels)
            
            #test random forest performance for accuracy
            accuracy = clf.score(test_data, test_labels)
            
            #test random forest performance for class probabilities
            class_probs = clf.predict_proba(test_data)
            class_labels = test_labels
            
            #add metrics to results dictionary
            crossval_accuracies.append(accuracy)
            
            #placeholder for storing class prediction/other metrics in crossval_metrics
    print("printing accuracies list: ")
    print(crossval_accuracies)
    print("num_labels = "+str(len(merged_arr)))
    print("subset size = "+str(len(merged_arr)/num_partitions))
    print("mean accuracy = "+str(sum(crossval_accuracies)/len(crossval_accuracies)))
    cluster_num_accuracies.append(sum(crossval_accuracies)/len(crossval_accuracies))
    cluster_num_sems.append(sem(crossval_accuracies))
"""

#plot accurracies by cluster number
print("cluster accuracies by num: 2-10")
print(cluster_num_accuracies)

x_coordinates = [i for i in range(2, cluster_test_numbers +1)]
plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
#plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, Model Parameters")
plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, Model Parameters")
plt.xlabel("Num Clusters")
plt.ylabel("Mean Cross-Val Accuracy")
plt.ylim((0.6,None))
plt.show()


#To use: random.shuffle()

#-------------------------------------------------------------------

#plot clusters
#axis_labels = ['mu_baseline', 'mu1', 'sigma_baseline']
legend_handles_cluster = []
legend_handles_type = []
axis_labels = ['pc-one', 'pc-two','pc-three']

for entry in marker_dict:
    label = entry
    marker = marker_dict[entry]
    examplar = mlines.Line2D([], [], color='blue', marker=marker,
                          markersize=15, linestyle="", label=label)
    legend_handles_type.append(examplar)

first_legend = plt.legend(handles = legend_handles_type, loc='upper right')
plt.gca().add_artist(first_legend)

for i in range(0, len(label_sorted)):
    label = "cluster "+str(i)
    colour = colours[i]
    colour_patch = mpatches.Patch(color=colour, label=label)
    legend_handles_cluster.append(colour_patch)
    #marker =  marker
    #ax = ax
    #print(cell_label_sorted)
    try:
        #print("try")
        dim1 = label_sorted[i][:, 0]
        dim2 = label_sorted[i][:, 1]
        dim3 = label_sorted[i][:, 2]
        #post_labels = post_label_sorted[i][:]
        post_labels = [row[1] for row in cell_label_sorted[i]]
        #print("post_labels = "+str(post_labels))
    except:
        #print("except")
        dim1 = label_sorted[i][0]
        dim2 = label_sorted[i][1]
        dim3 = label_sorted[i][2]
        post_labels = cell_label_sorted[i][0][1]
        #print("post_labels = "+str(post_labels))
    add_to_plot_3D(ax, label, colour, post_labels, dim1, dim2, dim3)
        
    
title_name = "Cluster_PCA_123_Standard_Acsf_sigmas"
x_label = axis_labels[0],
y_label = axis_labels[1],
z_label = axis_labels[2],

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
#plt.legend()
plt.legend(handles=legend_handles_cluster, loc='lower right')
plt.title(title_name)

plt.show()


#example plottting code for clusters
"""
plt.close('all')
plt.figure(1)
plt.clf()
"""
"""
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = scaled_arr[cluster_centers_indices[k]]
    plt.plot(scaled_arr[class_members, 0], scaled_arr[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in scaled_arr[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
"""
print('Estimated number of clusters: %d' % n_clusters_)
    
    
    
#actually run plot
desired_dims = [0,1,2]
pca_axis_labels =['pc-one', 'pc-two','pc-three']
title_pca = "3D_PCA_123_Standard_acsf_sigma_amps"
#working_plot1(desired_dims, pca_post_pre, NUM_COLORS, pca_axis_labels, title_pca)


title_raw = "3D_params_Standard_acsf"
desired_dims_raw = [0,1,4]
raw_axis_labels = ['mu_baseline', 'mu1', 'sigma_baseline']
#working_plot1(desired_dims_raw, raw_post_pre, NUM_COLORS, raw_axis_labels, title_raw)
            
