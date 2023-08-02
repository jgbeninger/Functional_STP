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
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.utils.random import sample_without_replacement


#these come from example code and may not be needed
import time
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
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

filtered_file = "removed_vals.txt"
filtered_out = open(filtered_file, "w")

num_100s = 0
for i in range(1, len(sys.argv)):
    cut_on = '_'
    file_name = sys.argv[i]
    print(i)
    print(file_name)
    pair_id = file_name.split(cut_on)[8] #6 for standard, 8 for two step, 9 for windows
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[6] #4 for standard, 6 for two step, 7 for windows
    pre_type = pre_type.split(cut_on)[0]
    post_type = file_name.split(cut_on)[7] #5 for standard, 7 for two step, 8 for windows
    post_type = post_type.split(cut_on)[0]
    
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
    
    print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+str(pair_id))
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
    print(params)
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
        print('attempted key = ')
        print((pre_type, post_type))
        print("keys = ")
        print(measures_dict[(pre_type, post_type)][pair_id].keys())
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        print("printing ppr:" +str(ppr))
        #skip the excessively high ppr
        
        
        if ppr > 4: #1000 for the excessive value
            print("skipper ppr = "+str(ppr))
            filtered_out.write(file_name)
            filtered_out.write("\n")
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

filtered_out.close()
#print(data)
num_cols = len(data[0])
#print(data[:,3:])
#params_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
#params_data = [row[3:] for row in data[:]] #model and phys
params_data = [row[8:] for row in data[:]] #model only

phys_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
#phys_data =[row[3:8]+[row[12]] for row in data[:]] #phys only w/ 50Hz rec
hybrid_data = [row[3:] for row in data[:]] #model and phys
model_data = [row[8:13] for row in data[:]] #model only
#model_data = [row[8:12] for row in data[:]] #only mu kernel data

#params_data = [row[3:8] for row in data[:]]
#params_data = [row[3:10] for row in data[:]]


#params_data = [row[8:] for row in data[:]]
#row_labels = [row[0:8] for row in data[:]]
row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)



#--------------------------------------------------------------------------

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

#test the top model prep configurations
"""
print("phys case (special) 2 clusters")
list_cluster_pairs(phys_data, row_labels, num_clusters=2)

print("model, 5 clusters")
list_cluster_pairs(model_data, row_labels, num_clusters=5)

print("hybrid case, 2 clusters")
list_cluster_pairs(hybrid_data, row_labels, num_clusters=2)
"""

def cre_accuracy(params, row_labels, cre_type= 'sst', pre_index =0, post_index =1, partition_size=20):
    raw_data = params.copy()

    
    #remove this section to run without scaling + pca
    pca = PCA(whiten=True)

    #scale data before pca
    scaler = StandardScaler()
    scaler.fit(raw_data)
    scaled_arr = scaler.transform(raw_data)
    pca_arr = pca.fit_transform(scaled_arr)
    data = pca_arr.tolist()
    
    test_accuracies = []
    baseline_accuracies = []
    num_of_type = 0.0
    num_total = 0.0
    #assign desired post cre_type as objective for binary classification
    for i in range(0, len(row_labels)):
        pre_type = row_labels[i][pre_index]
        post_type = row_labels[i][post_index]
        if post_type == cre_type:
            #targets.append(1)
            #add label to data before shuffling
            data[i].append(1)
            num_of_type += 1
        else:
            #targets.append(0)
            #add label to data before shuffling
            data[i].append(0)
        num_total += 1
    #randomly shuffle
    for i in range(0, 1000):
        arr_shuffled = data.copy()
        random.shuffle(arr_shuffled)
        
        #peform cross validation
        partitioned_list  = partition_list(arr_shuffled, partition_size)
        num_partitions = int(len(arr_shuffled)/partition_size)
        
        for n in range(0, num_partitions):
            
            #get train and test sets for this cross val step
            test_set = partitioned_list[n]
            #get all subsets for training set
            train_subsets = partitioned_list[:n]+partitioned_list[n+1:] #is this valid? What about edge cases?
            #append all training set subsets to create single 2D list
            train_set = [e for sbl in train_subsets for e in sbl] #also sanity check this
            
            #divide targets and train data
            #num_data = len(train_set)-1
            train_data = [row[0:(len(row)-1)] for row in train_set]
            train_targets = [row[len(row)-1] for row in train_set]
            
            #num_test = len(test_set)-1
            test_data = [row[0:(len(row)-1)] for row in test_set]
            test_targets = [row[len(row)-1] for row in test_set]
            
            num_type_in_test = len([1 for entry in test_targets if (abs(entry- 1) < 0.1)])
            
            print("num_type in test = "+str(num_type_in_test))
            print("len(test) = " +str(len(test_data)))
            print("printing test data")
            print(test_targets)

            """
            if num_type_in_test < len(test_targets )/2:
                baseline_accuracies.append((len(test_targets)-num_type_in_test)/len(test_targets))
            else:
                baseline_accuracies.append(num_type_in_test/len(test_targets))
            """
            
            
            #ensure both train and test sets contain at least one example of the class
            if not ((1 in test_targets) and (0 in test_targets)):
                #print("test data missing a class")
                #print(test_targets)
                continue
            if not ((1 in train_targets) and (0 in train_targets)):
                #print("train data missing a class")
                #print(train_targets)
                continue
            
            #test baseline accuracy on this partition
            dummy_clf = DummyClassifier(strategy='most_frequent')
            dummy_clf.fit(train_data, train_targets)
            baseline_accuracy = dummy_clf.score(test_data, test_targets)
            baseline_accuracies.append(baseline_accuracy)
        
            #test accuracy on this partition
            #clf = LogisticRegression(max_iter=10000)
            #clf = SVC()
            clf.fit(train_data, train_targets)
            accuracy = clf.score(test_data, test_targets)
            test_accuracies.append(accuracy)
            
    """
    clf = LogisticRegression(max_iter=10000)
    train_data = params[0:90]
    train_targets = targets[0:90]
    test_data = params[90:]
    test_targets = targets[90:]
    clf.fit(train_data, train_targets)
    accuracy = clf.score(test_data, test_targets)
    """
    return (test_accuracies, baseline_accuracies)
        

def cre_accuracy_multilabel(params, row_labels, pre_index =0, target_index =1, partition_size=10):
    raw_data = params.copy()

    
    #remove this section to run without scaling + pca
    pca = PCA(whiten=True)

    #scale data before pca

    scaler = StandardScaler()
    scaler.fit(raw_data)
    scaled_arr = scaler.transform(raw_data)
    pca_arr = pca.fit_transform(scaled_arr)
    data = pca_arr.tolist()

    
    #redefining to remove PCA for some alg tests
    #data=raw_data
    #pca_arr =scaled_arr
    #pca_arr = np.asarray(raw_data)
    
    test_accuracies = []
    baseline_accuracies = []
    num_of_type = 0.0
    num_total = 0.0
    #assign desired post cre_type as objective for binary classification
    #enc = MultiLabelBinarizer()
    enc = LabelEncoder()
    #enc = OneHotEncoder(categories = 'auto')
    print("printing categories")
    #print([[row[post_index], len(data[0])] for row in row_labels])
    categories = np.asarray([row[target_index] for row in row_labels])
    old_categories = categories.copy()
    #filter out rare classes
    delete_list = []
    """
    for i in range(0, len(old_categories)):
        print(categories[i])
        if old_categories[i] == 'vip':
            delete_list.append(i)
        if old_categories[i] == 'tlx3':
            delete_list.append(i)
        if old_categories[i] == 'fam84b':
            delete_list.append(i)
    categories = np.delete(categories, delete_list, axis=0)
    pca_arr = np.delete(pca_arr, delete_list, axis=0)
    """
    print(categories)
    
    one_hot_targets = enc.fit_transform(categories)
    categories = one_hot_targets
    print(categories)
    #one_hot_targets = enc.transform(categories) #create one hot encoding of the outputs
    np.set_printoptions(threshold=sys.maxsize)
    print("printing encodings")
    print(one_hot_targets)
    
    """
    for i in range(0, len(row_labels)):
        pre_type = row_labels[i][pre_index]
        post_type = row_labels[i][post_index]
        if post_type == cre_type:
            #targets.append(1)
            #add label to data before shuffling
            data[i].append(1)
            num_of_type += 1
        else:
            #targets.append(0)
            #add label to data before shuffling
            data[i].append(0)
        num_total += 1
    """
        
    """
    #randomly shuffle
    for i in range(0, 10):
        arr_shuffled = data.copy()
        random.shuffle(arr_shuffled)
        
        #peform cross validation
        partitioned_list  = partition_list(arr_shuffled, partition_size)
        num_partitions = int(len(arr_shuffled)/partition_size)
    """
    num_partitions = int(len(data)/partition_size)
    #print("data = ")
    #print(data)
    
    for i in range(0, 1000): 
        X_shuffle, y_shuffle = shuffle(pca_arr, categories)
        #print("X_shuffle")
        #print(X_shuffle)
        print("y_shuffle")
        print(y_shuffle)
        for n in range(1, num_partitions+1):
            
            """
            #get train and test sets for this cross val step
            test_set = partitioned_list[n]    
            #get all subsets for training set
            train_subsets = partitioned_list[:n]+partitioned_list[n+1:] #is this valid? What about edge cases?
            #append all training set subsets to create single 2D list
            train_set = [e for sbl in train_subsets for e in sbl] #also sanity check this
            """
            #try:
            #divide targets and train data
            #num_data = len(train_set)-1
            #train_data = [row[0:(len(row)-1)] for row in train_set]
            #train_targets = [row[len(row)-1] for row in train_set]
            train_data = np.append(X_shuffle[:(n-1)*partition_size, :], X_shuffle[n*partition_size:, :], axis=0)
            train_targets = np.append(y_shuffle[:(n-1)*partition_size], y_shuffle[n*partition_size:], axis=0)
            #print("train_data:")
            #print(train_data)
            
            #print("train_targets:")
            #print(train_targets)
            
            
            #num_test = len(test_set)-1
            #test_data = [row[0:(len(row)-1)] for row in test_set]
            #test_targets = [row[len(row)-1] for row in test_set]
            test_data = X_shuffle[(n-1)*partition_size:n*partition_size, :]
            test_targets = y_shuffle[(n-1)*partition_size:n*partition_size]
            #print("test_data:")
            #print(test_data)
            print("train_targets")
            print(train_targets)
            
            
            #num_type_in_test = len([1 for entry in test_targets if (abs(entry- 1) < 0.1)])
            
            """
            print("num_type in test = "+str(num_type_in_test))
            print("len(test) = " +str(len(test_data)))
            print("printing test data")
            print(test_targets)
            """
            
    
            """
            if num_type_in_test < len(test_targets )/2:
                baseline_accuracies.append((len(test_targets)-num_type_in_test)/len(test_targets))
            else:
                baseline_accuracies.append(num_type_in_test/len(test_targets))
            """
            
            
            """
            #ensure both train and test sets contain at least one example of the class
            if not ((1 in test_targets) and (0 in test_targets)):
                #print("test data missing a class")
                #print(test_targets)
                continue
            if not ((1 in train_targets) and (0 in train_targets)):
                #print("train data missing a class")
                #print(train_targets)
                continue
            """
        
            #test accuracy on this partition
            #clf = LinearSVC()
            #clf = LogisticRegression(max_iter=10000)
            clf = SVC()
            #clf = GradientBoostingClassifier(n_estimators=500)
            #clf = RF()
            #clf = AdaBoostClassifier()
            #clf = MLPClassifier(hidden_layer_sizes=(50,50,50))
            #multi_target_svc = MultiOutputClassifier(clf, n_jobs=-1)
            clf.fit(train_data, train_targets)
            accuracy = clf.score(test_data, test_targets)
            print("appending accuracy")
            print(accuracy)
            test_accuracies.append(accuracy)
            
            #test baseline accuracy on this partition
            dummy_clf = DummyClassifier(strategy='most_frequent')
            dummy_clf.fit(train_data, train_targets)
            baseline_accuracy = dummy_clf.score(test_data, test_targets)
            predictions = dummy_clf.predict(test_data)
            print("printing baseline predictions")
            print(predictions)
            baseline_accuracies.append(baseline_accuracy)
            print("printing baseline_accuracy")
            print(baseline_accuracy)
            
            """
            clf = LogisticRegression(max_iter=10000)
            train_data = params[0:90]
            train_targets = targets[0:90]
            test_data = params[90:]
            test_targets = targets[90:]
            clf.fit(train_data, train_targets)
            accuracy = clf.score(test_data, test_targets)
            """
            #except:
            #print("ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0")
    return (test_accuracies, baseline_accuracies)

def cre_accuracy_bootstrap(params, row_labels, pre_index =0, target_index =1, partition_size=10):
    raw_data = params.copy()

    
    #remove this section to run without scaling + pca
    pca = PCA(whiten=True)

    #artificially reduce sample size to match 2mM conditions
    """
    idx = np.random.randint(len(raw_data), size=78)
    raw_data = np.asarray(raw_data)
    raw_data = raw_data[idx, :]
    
    row_labels = np.asarray(row_labels)
    row_labels = row_labels[idx, :]
    """

    #scale data before pca

    scaler = StandardScaler()
    print("printing raw_data")
    print(raw_data)
    scaler.fit(raw_data)
    scaled_arr = scaler.transform(raw_data)
    pca_arr = pca.fit_transform(scaled_arr)
    data = pca_arr.tolist()

    
    #redefining to remove PCA for some alg tests
    #data=raw_data
    #pca_arr =scaled_arr
    #pca_arr = np.asarray(raw_data)
    
    test_accuracies = []
    baseline_accuracies = []
    differences = []
    num_of_type = 0.0
    num_total = 0.0
    #assign desired post cre_type as objective for binary classification
    #enc = MultiLabelBinarizer()
    enc = LabelEncoder()
    #enc = OneHotEncoder(categories = 'auto')
    print("printing categories")
    #print([[row[post_index], len(data[0])] for row in row_labels])
    categories = np.asarray([row[target_index] for row in row_labels])
    old_categories = categories.copy()
    #filter out rare classes
    delete_list = []
    print(categories)
    
    one_hot_targets = enc.fit_transform(categories)
    categories = one_hot_targets
    print(categories)
    #one_hot_targets = enc.transform(categories) #create one hot encoding of the outputs
    np.set_printoptions(threshold=sys.maxsize)
    print("printing encodings")
    print(one_hot_targets)
    
    num_partitions = int(len(data)/partition_size)
    #num_partitions = 10
    #print("data = ")
    #print(data)
    
    #remove single example category as it cannot be both trained and tested on:
    categories = np.delete(categories, [31])
    pca_arr = np.delete(pca_arr, [31], axis=0)
    
    alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm"]
    alg_outputs = [[-1 for i in range(0, 3)] for j in range(0, len(alg_labels))]
    
    for i in range(0, 3000): 

        #artificially reduce sample size to match 2mM conditions
        #idx = np.random.randint(len(pca_arr), size=78, replace=False)

        """
        idx = np.random.choice(len(pca_arr), size=67, replace=False)
        #print("idx:")
        #print(idx)
        pca_arr_subsampled = pca_arr[idx, :]
        print("categories unmodified:")
        print(categories)
        categories_subsampled = np.asarray([categories[i] for i in idx])
        print("categories subsampled:")
        print(categories_subsampled)
        print("len(categories):")
        print(len(categories))
        """

        #redefine num partitions:
        num_partitions = int(len(pca_arr)/partition_size)

        """
        categories_subsampled = categories


        print("categories")
        print(categories)
        """
        
        #try:
        #X_shuffle, y_shuffle = resample(pca_arr, categories) #with replacement
        #X_shuffle, y_shuffle = shuffle(pca_arr, categories) #without replacement
        X_shuffle, y_shuffle = shuffle(pca_arr, categories) #without replacement but with subsampling
        #print("X_shuffle")
        #print(X_shuffle)
        #print("y_shuffle")
        #print(y_shuffle)
        cross_accuracies = [[] for i in range(0, len(alg_labels))]
        cross_baselines = [[] for i in range(0, len(alg_labels))]
        for n in range(1, num_partitions+1):
            
            #try:
            #divide targets and train data
            #num_data = len(train_set)-1
            #train_data = [row[0:(len(row)-1)] for row in train_set]
            #train_targets = [row[len(row)-1] for row in train_set]
            train_data = np.append(X_shuffle[:(n-1)*partition_size, :], X_shuffle[n*partition_size:, :], axis=0)
            train_targets = np.append(y_shuffle[:(n-1)*partition_size], y_shuffle[n*partition_size:], axis=0)
            #print("train_data:")
            #print(train_data)
            
            #print("train_targets:")
            #print(train_targets)
            
            
            #num_test = len(test_set)-1
            #test_data = [row[0:(len(row)-1)] for row in test_set]
            #test_targets = [row[len(row)-1] for row in test_set]
            test_data = X_shuffle[(n-1)*partition_size:n*partition_size, :]
            test_targets = y_shuffle[(n-1)*partition_size:n*partition_size]
            print("test_data:")
            print(test_data)
            print("test_targets")
            print(test_targets)
            
            
            #num_type_in_test = len([1 for entry in test_targets if (abs(entry- 1) < 0.1)])
        
            #test accuracy on this partition
            #clf = LinearSVC()
            clf_lr = LogisticRegression(max_iter=10000)
            clf_svm = SVC()
            clf_gb = GradientBoostingClassifier(n_estimators=500)
            clf_rf = RF()
            clf_ad = AdaBoostClassifier()
            clf_mlp = MLPClassifier(hidden_layer_sizes=(50,50,50))
            clf_list =  [clf_gb, clf_lr, clf_ad, clf_mlp, clf_rf, clf_svm]
            for index in range(0, len(clf_list)):
                
                clf = clf_list [index]
                #multi_target_svc = MultiOutputClassifier(clf, n_jobs=-1)
                clf.fit(train_data, train_targets)
                accuracy = clf.score(test_data, test_targets)
                print("appending accuracy")
                print(accuracy)
                
                #test baseline accuracy on this partition
                dummy_clf = DummyClassifier(strategy='most_frequent')
                dummy_clf.fit(train_data, train_targets)
                baseline_accuracy = dummy_clf.score(test_data, test_targets)
                predictions = dummy_clf.predict(test_data)
                
                #print(index)
                #alg_outputs[index][0].append(accuracy)
                #alg_outputs[index][1].append(baseline_accuracy)
                cross_accuracies[index].append(accuracy)
                #test_accuracies.append(accuracy)
                
                #print("printing baseline predictions")
                #print(predictions)
                #baseline_accuracies.append(baseline_accuracy)
                cross_baselines[index].append(baseline_accuracy)
                #print("printing baseline_accuracy")
                #print(baseline_accuracy)
                #differences.append(accuracy-baseline_accuracy)
        
            """
            #skip random partition if a category has too few examplars for classification
            except:
                print("ValueError: The number of classes has to be greater than one; got 1 class")
                continue
            """
        for i in range(0, len(alg_labels)):
            accuracy = statistics.mean(cross_accuracies[i])
            baseline = statistics.mean(cross_baselines[i])
            difference = accuracy - baseline 
            
            if alg_outputs[index][0] == -1: #if not filled
                alg_outputs[i][0] = [accuracy]
                alg_outputs[i][1] = [baseline]
                alg_outputs[i][2] = [difference]
                print("alg_outputs[i][0] = [accuracy]")
                print(alg_outputs[i][0])
                    
            else:
                print(alg_outputs[i][0])
                alg_outputs[i][0].append(accuracy)
                alg_outputs[i][1].append(baseline)
                alg_outputs[i][2].append(difference)
        
            #except:
            #print("ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0")
    #return (test_accuracies, baseline_accuracies, differences)
    return (alg_outputs, alg_labels)


def cre_accuracy_bootstrap_single_split(params, row_labels, pre_index =0, target_index =1, partition_size=10):
    raw_data = params.copy()

    
    #remove this section to run without scaling + pca
    pca = PCA(whiten=True)

    #scale data before pca

    scaler = StandardScaler()
    scaler.fit(raw_data)
    scaled_arr = scaler.transform(raw_data)
    pca_arr = pca.fit_transform(scaled_arr)
    data = pca_arr.tolist()

    
    #redefining to remove PCA for some alg tests
    #data=raw_data
    #pca_arr =scaled_arr
    #pca_arr = np.asarray(raw_data)
    
    test_accuracies = []
    baseline_accuracies = []
    differences = []
    num_of_type = 0.0
    num_total = 0.0
    #assign desired post cre_type as objective for binary classification
    #enc = MultiLabelBinarizer()
    enc = LabelEncoder()
    #enc = OneHotEncoder(categories = 'auto')
    print("printing categories")
    #print([[row[post_index], len(data[0])] for row in row_labels])
    categories = np.asarray([row[target_index] for row in row_labels])
    old_categories = categories.copy()
    #filter out rare classes
    delete_list = []
    print(categories)
    
    one_hot_targets = enc.fit_transform(categories)
    categories = one_hot_targets
    
    #remove categories without at leas two samples (to allow for train and test)
    #remove the only single examplar of a class: '3', number 31
    categories = np.delete(categories, [31])
    pca_arr = np.delete(pca_arr, [31], axis=0)
    print(categories)
    #one_hot_targets = enc.transform(categories) #create one hot encoding of the outputs
    np.set_printoptions(threshold=sys.maxsize)
    print("printing encodings")
    print(one_hot_targets)
    
    num_partitions = int(len(data)/partition_size)
    #num_partitions = 10
    #print("data = ")
    #print(data)
    
    for i in range(0, 10000): 
        #X_shuffle, y_shuffle = resample(pca_arr, categories) #with replacement
        print(categories)
        train_data, test_data, train_targets, test_targets = train_test_split(pca_arr, categories, stratify=categories, test_size=0.35)
        #X_shuffle, y_shuffle = shuffle(pca_arr, categories) #without replacement
        #print("X_shuffle")
        #print(X_shuffle)
        #print("y_shuffle")
        #print(y_shuffle)

        #try:
        #divide targets and train data
        #num_data = len(train_set)-1
        #train_data = [row[0:(len(row)-1)] for row in train_set]
        #train_targets = [row[len(row)-1] for row in train_set]
        #train_data = np.append(X_shuffle[:(n-1)*partition_size, :], X_shuffle[n*partition_size:, :], axis=0)
        #train_targets = np.append(y_shuffle[:(n-1)*partition_size], y_shuffle[n*partition_size:], axis=0)
        #print("train_data:")
        #print(train_data)
        
        #print("train_targets:")
        #print(train_targets)
        
        
        #num_test = len(test_set)-1
        #test_data = [row[0:(len(row)-1)] for row in test_set]
        #test_targets = [row[len(row)-1] for row in test_set]
        #test_data = X_shuffle[(n-1)*partition_size:n*partition_size, :]
        #test_targets = y_shuffle[(n-1)*partition_size:n*partition_size]
        #print("test_data:")
        #print(test_data)
        print("train_targets")
        print(train_targets)
        
        
        #num_type_in_test = len([1 for entry in test_targets if (abs(entry- 1) < 0.1)])
    
        #test accuracy on this partition
        #clf = LinearSVC()
        #clf = LogisticRegression(max_iter=10000)
        clf = SVC()
        #clf = GradientBoostingClassifier(n_estimators=500)
        #clf = RF()
        #clf = AdaBoostClassifier()
        #clf = MLPClassifier(hidden_layer_sizes=(50,50,50))
        #multi_target_svc = MultiOutputClassifier(clf, n_jobs=-1)
        clf.fit(train_data, train_targets)
        accuracy = clf.score(test_data, test_targets)
        print("appending accuracy")
        print(accuracy)
        #test_accuracies.append(accuracy)
        
        #test baseline accuracy on this partition
        dummy_clf = DummyClassifier(strategy='most_frequent')
        dummy_clf.fit(train_data, train_targets)
        baseline_accuracy = dummy_clf.score(test_data, test_targets)
        predictions = dummy_clf.predict(test_data)
        print("printing baseline predictions")
        print(predictions)
        #baseline_accuracies.append(baseline_accuracy)
        print("printing baseline_accuracy")
        print(baseline_accuracy)
        #differences.append(accuracy-baseline_accuracy)
        test_accuracies.append(accuracy)
        baseline_accuracies.append(baseline_accuracy)
        difference = accuracy - baseline_accuracy
        differences.append(difference)
        
        """
        accuracy = statistics.mean(cross_accuracies)
        baseline = statistics.mean(cross_baselines)
        difference = accuracy - baseline 
        
        test_accuracies.append(accuracy)
        baseline_accuracies.append(baseline)
        differences.append(difference)
        """
        
            #except:
            #print("ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0")
    return (test_accuracies, baseline_accuracies, differences)

#paired testing
"""
print("printing accuracies")
print("phys case (special) 2 clusters, sst accuracy")
phys_sst_accuracies, phys_sst_baseline = cre_accuracy(phys_data, row_labels)
phys_sst_mean = statistics.mean(phys_sst_accuracies)
phys_sst_sem = stat.sem(np.array(phys_sst_accuracies))
phys_sst_pval = wilcoxon(phys_sst_accuracies, phys_sst_baseline).pvalue

print("mean accuracy = "+str(phys_sst_mean) + " sem = "+str(phys_sst_sem))
print("baseline = "+str(phys_sst_baseline))

#print("accuracy = "+str(cre_accuracy(model_data, row_labels)))

print("model, 5 clusters sst accuracy")
model_sst_accuracies, model_sst_baseline = cre_accuracy(model_data, row_labels)
model_sst_mean = statistics.mean(model_sst_accuracies)
model_sst_sem = stat.sem(np.array(model_sst_accuracies))
model_sst_pval = wilcoxon(model_sst_accuracies, model_sst_baseline).pvalue

print("mean accuracy = "+str(model_sst_mean) + " sem = "+str(model_sst_sem))
print("baseline = "+str(model_sst_baseline))

#print("hybrid case, 2 clusters sst accuracy")
#print("accuracy = "+str(cre_accuracy(hybrid_data, row_labels)))

print("hybrid case, 2 clusters sst accuracy")
hybrid_sst_accuracies, hybrid_sst_baseline = cre_accuracy(hybrid_data, row_labels)
hybrid_sst_mean = statistics.mean(hybrid_sst_accuracies)
hybrid_sst_sem = stat.sem(np.array(hybrid_sst_accuracies))
hybrid_sst_pval = wilcoxon(hybrid_sst_accuracies, hybrid_sst_baseline).pvalue

sst_baseline_mean = statistics.mean(hybrid_sst_baseline)
sst_baseline_sem = stat.sem(np.array(hybrid_sst_baseline))
print("mean accuracy = "+str(hybrid_sst_mean) + " sem = "+str(hybrid_sst_sem))
print("baseline = "+str(hybrid_sst_baseline))

print("printing accuracies")
#print("phys case (special) 2 clusters, pvalb accuracy")
#print("accuracy = "+str(cre_accuracy(phys_data, row_labels, cre_type='pvalb')))

print("phys case (special) 2 clusters, pvalb accuracy")
phys_pv_accuracies, phys_pv_baseline = cre_accuracy(phys_data, row_labels, cre_type='pvalb')
phys_pv_mean = statistics.mean(phys_pv_accuracies)
phys_pv_sem = stat.sem(np.array(phys_pv_accuracies))
phys_pv_pval = wilcoxon(phys_pv_accuracies, phys_pv_baseline).pvalue
print("mean accuracy = "+str(phys_pv_mean) + " sem = "+str(phys_pv_sem))
print("baseline = "+str(phys_pv_baseline))

#print("model, 5 clusters pvalb accuracy")
#print("accuracy = "+str(cre_accuracy(model_data, row_labels, cre_type='pvalb')))
print("model, 5 clusters pvalb accuracy")
model_pv_accuracies, model_pv_baseline  = cre_accuracy(model_data, row_labels, cre_type='pvalb')
model_pv_mean = statistics.mean(model_pv_accuracies)
model_pv_sem = stat.sem(np.array(model_pv_accuracies))
model_pv_pval = wilcoxon(model_pv_accuracies, model_pv_baseline).pvalue
print("mean accuracy = "+str(model_pv_mean) + " sem = "+str(model_pv_sem))
print("baseline = "+str(model_pv_baseline))

#print("hybrid case, 2 clusters pvalb accuracy")
#print("accuracy = "+str(cre_accuracy(hybrid_data, row_labels, cre_type='pvalb')))
print("hybrid case, 2 clusters pvalb accuracy")
hybrid_pv_accuracies, hybrid_pv_baseline = cre_accuracy(hybrid_data, row_labels, cre_type='pvalb')
hybrid_pv_mean = statistics.mean(hybrid_pv_accuracies)
hybrid_pv_sem = stat.sem(np.array(hybrid_pv_accuracies))
hybrid_pv_pval = wilcoxon(hybrid_pv_accuracies, hybrid_pv_baseline).pvalue

pv_baseline_mean = statistics.mean(hybrid_pv_baseline)
pv_baseline_sem = stat.sem(np.array(hybrid_pv_baseline))
print("mean accuracy = "+str(hybrid_pv_mean) + " sem = "+str(hybrid_pv_sem))
print("baseline = "+str(hybrid_pv_baseline))

#x_labels = ["phys_sst \n","phys_pv \n", "model_sst \n", "model_pv \n", "hybrid_sst \n", "hybrid_pv \n", "sst_baseline", "pv_baseline"]
x_labels = ["phys_sst \n"+pval_to_str(phys_sst_pval),"phys_pv \n"+pval_to_str(phys_pv_pval), "model_sst \n"+pval_to_str(model_sst_pval), "model_pv \n"+pval_to_str(model_pv_pval), "hybrid_sst \n"+pval_to_str(hybrid_sst_pval), "hybrid_pv \n"+pval_to_str(hybrid_pv_pval), "sst_baseline", "pv_baseline"]
y_vals = [phys_sst_mean, phys_pv_mean, model_sst_mean, model_pv_mean, hybrid_sst_mean, hybrid_pv_mean, sst_baseline_mean, pv_baseline_mean]
y_errors = [phys_sst_sem, phys_pv_sem, model_sst_sem, model_pv_sem, hybrid_sst_sem, hybrid_pv_sem, sst_baseline_sem, pv_baseline_sem]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.title("Supervised Classification Accuracy By Representation With PCA")
plt.xticks(x_coordinates, x_labels)
plt.xlabel("Representation")
plt.ylabel("Mean Accuracy")
plt.ylim([0,1])
plt.show()
"""

"""
#multilabel classification (post):
    
phys_multi_accuracies, phys_multi_baseline = cre_accuracy_multilabel(phys_data, row_labels)
phys_multi_mean = statistics.mean(phys_multi_accuracies)
phys_multi_sem = stat.sem(np.array(phys_multi_accuracies))

phys_baseline_mean = statistics.mean(phys_multi_baseline)
phys_baseline_sem = stat.sem(np.array(phys_multi_baseline))
phys_pval = wilcoxon(phys_multi_accuracies, phys_multi_baseline).pvalue

model_multi_accuracies, model_multi_baseline = cre_accuracy_multilabel(model_data, row_labels)
model_multi_mean = statistics.mean(model_multi_accuracies)
model_multi_sem = stat.sem(np.array(model_multi_accuracies))

model_baseline_mean = statistics.mean(model_multi_baseline)
model_baseline_sem = stat.sem(np.array(model_multi_baseline))
model_pval = wilcoxon(model_multi_accuracies, model_multi_baseline).pvalue

hybrid_multi_accuracies, hybrid_multi_baseline = cre_accuracy_multilabel(hybrid_data, row_labels)
hybrid_multi_mean = statistics.mean(hybrid_multi_accuracies)
hybrid_multi_sem = stat.sem(np.array(hybrid_multi_accuracies))

hybrid_baseline_mean = statistics.mean(hybrid_multi_baseline)
hybrid_baseline_sem = stat.sem(np.array(hybrid_multi_baseline))
hybrid_pval = wilcoxon(hybrid_multi_accuracies, hybrid_multi_baseline).pvalue
"""

#
"""
#Alternates
phys_multi_accuracies, phys_multi_baseline, phys_multi_differences = cre_accuracy_bootstrap(phys_data, row_labels) #target_index=1 is post type
#phys_multi_accuracies, phys_multi_baseline, phys_multi_differences = cre_accuracy_bootstrap_single_split(phys_data, row_labels)
phys_multi_mean = statistics.mean(phys_multi_accuracies)
phys_multi_sem = stat.sem(np.array(phys_multi_accuracies))

phys_baseline_mean = statistics.mean(phys_multi_baseline)
phys_baseline_sem = stat.sem(np.array(phys_multi_baseline))
phys_pval = wilcoxon(phys_multi_accuracies, phys_multi_baseline).pvalue

model_multi_accuracies, model_multi_baseline, model_multi_differences = cre_accuracy_bootstrap(model_data, row_labels)
#model_multi_accuracies, model_multi_baseline, model_multi_differences = cre_accuracy_bootstrap_single_split(model_data, row_labels)
model_multi_mean = statistics.mean(model_multi_accuracies)
model_multi_sem = stat.sem(np.array(model_multi_accuracies))

model_baseline_mean = statistics.mean(model_multi_baseline)
model_baseline_sem = stat.sem(np.array(model_multi_baseline))
model_pval = wilcoxon(model_multi_accuracies, model_multi_baseline).pvalue


#pickle the outputs

phys_post_distributions = [phys_multi_accuracies, phys_multi_baseline, phys_multi_differences]
with open('./Figures/dists/physpostbootstrapdists1.3mMfull.p', 'wb') as handle:
    pickle.dump(phys_post_distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)

model_post_distributions = [model_multi_accuracies, model_multi_baseline, model_multi_differences]
with open('./Figures/dists/modelpostbootstrapdists1.3mMfull.p', 'wb') as handle:
    pickle.dump(model_post_distributions, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""


"""
hybrid_multi_accuracies, hybrid_multi_baseline = cre_accuracy_bootstrap(hybrid_data, row_labels)
hybrid_multi_mean = statistics.mean(hybrid_multi_accuracies)
hybrid_multi_sem = stat.sem(np.array(hybrid_multi_accuracies))

hybrid_baseline_mean = statistics.mean(hybrid_multi_baseline)
hybrid_baseline_sem = stat.sem(np.array(hybrid_multi_baseline))
hybrid_pval = wilcoxon(hybrid_multi_accuracies, hybrid_multi_baseline).pvalue


group_pval = kruskal(phys_multi_accuracies, model_multi_accuracies, hybrid_multi_accuracies).pvalue
"""

"""
#assume normality, homoscedascity
#group_pval = f_oneway(phys_multi_accuracies, model_multi_accuracies, hybrid_multi_accuracies).pvalue
print("test group val normality: phys, model, hybrid")
print(normaltest(phys_multi_accuracies))
print(normaltest(model_multi_accuracies))
#print(normaltest(hybrid_multi_accuracies))


#x_labels = ["phys_multi \n"+pval_to_str(phys_pval),"phys_baseline", "model_multi \n"+pval_to_str(model_pval), "model_baseline", "hybrid_multi \n"+pval_to_str(hybrid_pval), "hybrid_baseline"]
#y_vals = [phys_multi_mean, phys_baseline_mean, model_multi_mean, model_baseline_mean, hybrid_multi_mean, hybrid_baseline_mean]
#y_errors = [phys_multi_sem, phys_baseline_sem, model_multi_sem, model_baseline_sem, hybrid_multi_sem, hybrid_baseline_sem]
#x_labels = ["phys \n"+pval_to_str(phys_pval),"model \n"+pval_to_str(model_pval), "baseline"]
x_labels = ["phys","model", "baseline"]
y_vals  = [phys_multi_accuracies, model_multi_accuracies, model_multi_baseline]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Post Type Accuracy")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Mean Accuracy")
#plt.ylim([0,1])
save_name = "./Figures/fig3/multi_label_post_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
#plt.show()

#--------------------------------------------------
#repeat as bar plot
x_labels = ["phys","model", "baseline"]
y_vals  = [statistics.mean(phys_multi_accuracies), statistics.mean(model_multi_accuracies), statistics.mean(model_multi_baseline)]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
#medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.bar(x_coordinates, y_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Post Type Accuracy")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Mean Accuracy")
#plt.ylim([0,1])
save_name = "./Figures/fig3/multi_label_post_bootstrap_barv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
#plt.show()


#plot histogram of differences
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in model_multi_differences if entry>0])/len(model_multi_differences)
frac_nonneg = len([1 for entry in model_multi_differences if entry>=0])/len(model_multi_differences)
plt.hist(model_multi_differences, bins=30)
plt.title("Post_type model vs baseline histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))
save_name = "./Figures/fig3/multi_label_post_bootstrapv2_histmodel_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
#plt.show()


#plot histogram of differences
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in phys_multi_differences if entry>0])/len(phys_multi_differences)
frac_nonneg = len([1 for entry in phys_multi_differences if entry>=0])/len(phys_multi_differences)
plt.hist(phys_multi_differences, bins=30)
plt.title("Post_type phys vs baseline histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))
save_name = "./Figures/fig3/multi_label_post_bootstrapv2_histphys_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
#plt.show()
"""


#repeat for pre_type accuracy

"""
phys_multi_accuracies, phys_multi_baseline = cre_accuracy_multilabel(phys_data, row_labels, target_index =0)
phys_multi_mean = statistics.mean(phys_multi_accuracies)
phys_multi_sem = stat.sem(np.array(phys_multi_accuracies))

phys_baseline_mean = statistics.mean(phys_multi_baseline)
phys_baseline_sem = stat.sem(np.array(phys_multi_baseline))
phys_pval = wilcoxon(phys_multi_accuracies, phys_multi_baseline).pvalue

model_multi_accuracies, model_multi_baseline = cre_accuracy_multilabel(model_data, row_labels, target_index =0)
model_multi_mean = statistics.mean(model_multi_accuracies)
model_multi_sem = stat.sem(np.array(model_multi_accuracies))

model_baseline_mean = statistics.mean(model_multi_baseline)
model_baseline_sem = stat.sem(np.array(model_multi_baseline))
model_pval = wilcoxon(model_multi_accuracies, model_multi_baseline).pvalue
"""

"""
phys_alg_outputs, phys_alg_labels =cre_accuracy_bootstrap(phys_data, row_labels, target_index =0)
phys_multi_accuracies, phys_multi_baseline, phys_multi_differences = cre_accuracy_bootstrap(phys_data, row_labels, target_index =0)
phys_multi_mean = statistics.mean(phys_multi_accuracies)
phys_multi_sem = stat.sem(np.array(phys_multi_accuracies))

phys_baseline_mean = statistics.mean(phys_multi_baseline)
phys_baseline_sem = stat.sem(np.array(phys_multi_baseline))
phys_pval = wilcoxon(phys_multi_accuracies, phys_multi_baseline).pvalue

model_multi_accuracies, model_multi_baseline, model_multi_differences = cre_accuracy_bootstrap(model_data, row_labels, target_index =0)
model_multi_mean = statistics.mean(model_multi_accuracies)
model_multi_sem = stat.sem(np.array(model_multi_accuracies))

model_baseline_mean = statistics.mean(model_multi_baseline)
model_baseline_sem = stat.sem(np.array(model_multi_baseline))
model_pval = wilcoxon(model_multi_accuracies, model_multi_baseline).pvalue
"""

phys_alg_outputs, phys_alg_labels =cre_accuracy_bootstrap(phys_data, row_labels, target_index =1)
model_alg_outputs, phys_alg_labels =cre_accuracy_bootstrap(model_data, row_labels, target_index =1)

#pickle the outputs

with open('./Figures/dists3/physpostalgcompdists1.3mMfulln3000r2.p', 'wb') as handle:
    pickle.dump((phys_alg_outputs, phys_alg_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./Figures/dists3/modelpostalgcompdists1.3mMfulln3000r2.p', 'wb') as handle:
    pickle.dump((model_alg_outputs, phys_alg_labels), handle, protocol=pickle.HIGHEST_PROTOCOL)



#x_labels = ["phys\n"+pval_to_str(phys_pval),"model\n"+pval_to_str(model_pval), "baseline"]
x_labels = ["phys","model", "baseline"]
y_vals  = [phys_multi_accuracies, model_multi_accuracies, model_multi_baseline]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Pre Type Accuracy")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Mean Accuracy")
#plt.ylim([0,1])
#save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
f.set_size_inches(((2.3622, 2.465)))
f.set_dpi(1200)
f.tight_layout()
#plt.savefig(save_name, Transparent=True)
#plt.show()


#--------------------------------------------------
#repeat as bar plot

x_labels = ["phys","model", "baseline"]
y_vals  = [statistics.mean(phys_multi_accuracies), statistics.mean(model_multi_accuracies), statistics.mean(model_multi_baseline)]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.bar(x_coordinates, y_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Pre Type Accuracy")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Mean Accuracy")
#plt.ylim([0,1])
save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_bar_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_bar_1.3mM.svg"
f.set_size_inches(((2.3622, 2.465)))
f.set_dpi(1200)
f.tight_layout()
#plt.savefig(save_name, Transparent=True)
#plt.show()


#plot histogram of differences
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in model_multi_differences if entry>0])/len(model_multi_differences)
frac_nonneg = len([1 for entry in model_multi_differences if entry>=0])/len(model_multi_differences)
plt.hist(model_multi_differences, bins=30)
plt.title("Pre_type model vs baseline histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_histmodel_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
#plt.savefig(save_name, Transparent=True)

#plt.show()


#plot histogram of differences
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in phys_multi_differences if entry>0])/len(phys_multi_differences)
frac_nonneg = len([1 for entry in phys_multi_differences if entry>=0])/len(phys_multi_differences)
plt.hist(phys_multi_differences, bins=30)
plt.title("Pre_type phys vs baseline histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_histphys_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
#plt.savefig(save_name, Transparent=True)

#plt.show()

#--------------------------------------------------------------------------
pca = PCA(whiten=True)

#scale data before pca
scaler = StandardScaler()
"""
scaler.fit(params_arr)
scaled_arr = scaler.transform(params_arr)
"""
scaler.fit(model_data)
scaled_arr = scaler.transform(model_data)
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
"""
plt.hist(pca.explained_variance_ratio_)
plt.title(" Binned PCA Explained Variance Ratios By Component")
plt.xlabel("Explained variance ratio")
plt.ylabel("Num PCs")
plt.show()
"""

x_coordinates = [i for i in range(1, len(pca.explained_variance_ratio_)+1)]
plt.bar(x_coordinates, pca.explained_variance_ratio_)
plt.title("PCA Explained Variance Ratios By Component, model, 100Hz included "+str(len(model_data[0]))+" initial params")
plt.xlabel("PCA Component")
plt.ylabel("Explained Variance Ratio")
#plt.show()

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
NUM_COLORS = 5
plotted_points = 0
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
#plot function
"""
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
"""
desired_dims = [0,1,2]
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
    #plt.show()

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


km = KMeans(n_clusters=3, algorithm='full' ).fit(plot_arr)
#km = KMeans(n_clusters=3, algorithm='full' ).fit(scaled_arr)
#cluster_centers_indices = km.cluster_centers_indices_
labels = km.labels_

"""
ms = MeanShift(n_clusters=3).fit(plot_arr)
#cluster_centers_indices = km.cluster_centers_indices_
labels = ms.labels_
"""

#--------------------------------------------------------------------

#sort data by cluster

label_sorted = {}
cell_label_sorted = {}
#plot_arr = pca_result
#print("len(labels) = "+str(len(labels)))
#print(labels)
print("len(labels) = "+str(len(labels)))
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

#cross val for cluster stability measures: trained accuracy by split

#0 data struct to store pair cluster responses, do I need these pairs at first?
#1 random split process
#2 cross val loop with splits for accuracy measure, track also which pairs go where
num_partitions = 10
num_row_labels = 4
cluster_test_numbers = 15
cluster_num_accuracies = []
cluster_num_sems = []

def test_cre_clusters(labeled_arr, cre_types, cre_entry_index, num_clusters, cre_title, pre_proc_condition ):
    """
    labeled_arr: array with cluster labels as per merged_arr below
    
    cre_types: set of unique cre_types to check
    
    cre_entry_index: index in arr of cre_type to test against
    """
    
    #plotting colours
    NUM_COLORS = 10
    #colours = plt.cm.brg(np.linspace(0,1,NUM_COLORS))
    colours = plt.cm.tab10(np.linspace(0,1,NUM_COLORS))
    
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
            if (i % 2) == 0: #set alternating hashing
                rects1 = ax.bar(j -(width*num_clusters/2) + (i+0.5)*width, value, width, label=str(i), color = colours[i], hatch='/')
            else:
                rects1 = ax.bar(j -(width*num_clusters/2) + (i+0.5)*width, value, width, label=str(i), color = colours[i], hatch='\\')
    
    x = np.arange(len(cre_types)) 
    ax.set_xticks(x)
    ax.set_xticklabels(cre_types)
    
    #plot 
    #plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
    title  = "Number by cluster of cre "+cre_title + pre_proc_condition + " for "+str(num_clusters)+" clusters" 
    plt.title(title)
    plt.ylabel("Num pairs")
    #plt.show()
    
#get cluster accuracy
#0 data struct to store pair cluster responses, do I need these pairs at first?
#1 random split process
#2 cross val loop with splits for accuracy measure, track also which pairs go where
num_partitions = 10
num_row_labels = 4
cluster_test_numbers = 15
cluster_num_accuracies = []
cluster_num_sems = []
cluster_num_norm_scores = []
cluster_num_accuracy_dists = []

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
    cluster_num_accuracy_dists.append(crossval_accuracies)
    cluster_num_sems.append(sem(crossval_accuracies))
    #normality_value = normaltest(crossval_accuracies).pvalue
    #normality_value = shapiro(crossval_accuracies).pvalue
    #cluster_num_norm_scores.append(round(normality_value, 10))

#plot accurracies by cluster number
print("cluster accuracies by num: 2-10")
print(cluster_num_accuracies)

x_coordinates = [i for i in range(2, cluster_test_numbers +1)]

#plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
#plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, Model Parameters")

#calculate pairwise significance tests
pair_tests = ["NA"]
for i in range(0, len(cluster_num_accuracy_dists)-1):
    pval = wilcoxon(cluster_num_accuracy_dists[i], cluster_num_accuracy_dists[i+1]).pvalue
    pair_tests.append(round(pval, 2))
#plt.xlabel("Num Clusters")
#added for x-sublabels
f, ax = plt.subplots()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(x_coordinates)
ax.set_xticklabels(pair_tests)
#ax.set_xticklabels(cluster_num_norm_scores)

plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, SRP Model 2mM stand in")

plt.xlabel("P-value of comparison with left side column for cluster numbers 2-15")
plt.ylabel("Mean Cross-Val Accuracy")
plt.ylim((0.6,None))
f.set_size_inches((7.4, 2.54))
f.set_dpi(600)
f.tight_layout()
plt.legend()
save_name = "./Figures/fig4/stabilityv2_1.3mM.svg"
#plt.savefig(save_name)
#plt.show()
    
    
    
    
#below, commented out is code to show cre mappings, refactor to work with accuracy code above
#as functions

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
    pre_proc_condition = "PCA, Phys Only"
    test_cre_clusters(merged_arr, unique_pre, cre_entry_index, j, cre_title, pre_proc_condition)
    
    #repeat the operations below for ten random subset generation repetitions
    for i in range(0, 10):
    
        #divide data into ten random subsets without replacement
        arr_shuffled = merged_arr.copy()
        random.shuffle(arr_shuffled)
    
        
        partitioned_list  = partition_list(arr_shuffled, num_partitions)
        print("printing partitioned list")
        print(partitioned_list)
        for n in range(0, num_partitions):
            print("n = "+str(n))
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

#plot accurracies by cluster number
print("cluster accuracies by num: 2-10")
print(cluster_num_accuracies)

x_coordinates = [i for i in range(2, cluster_test_numbers +1)]
plt.bar(x_coordinates, cluster_num_accuracies, yerr=cluster_num_sems)
#plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, Phys Rec50")
plt.title("Cross Validation Accuracy of Different Cluster Numbers, PCA, Phys and Model")
plt.xlabel("Num Clusters")
plt.ylabel("Mean Cross-Val Accuracy")
plt.ylim((0.6,None))
plt.show()
"""

#--------------------------------------------------------------------

#plot clusters
#axis_labels = ['mu_baseline', 'mu1', 'sigma_baseline']
legend_handles_cluster = []
legend_handles_type = []
axis_labels = ['PPr_50Hz', 'Normed_areas_50Hz','Release_prob_all']
axis_labels = ['PPr_50Hz', 'Recoveries_50Hz','Release_prob_all']
#axis_labels = ['Component 1', 'Component 2','Component 3']

for entry in marker_dict:
    label = entry
    marker = marker_dict[entry]
    examplar = mlines.Line2D([], [], color='blue', marker=marker,
                          markersize=15, linestyle="", label=label)
    legend_handles_type.append(examplar)

first_legend = plt.legend(handles = legend_handles_type, loc='upper right')
plt.gca().add_artist(first_legend)

num_added = 0
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
        
        pprs = [row[5] for row in cell_label_sorted[i]]
        #areas = [row[4] for row in cell_label_sorted[i]]
        release_probs = [row[4] for row in cell_label_sorted[i]]
        recoveries= [row[7] for row in cell_label_sorted[i]]
        print("pprs =")
        print(pprs)
        #print("post_labels = "+str(post_labels))
    except:
        print("except")
        dim1 = label_sorted[i][0]
        dim2 = label_sorted[i][1]
        dim3 = label_sorted[i][2]
        post_labels = cell_label_sorted[i][0][1]
        
        pprs = cell_label_sorted[i][0][3]
        areas = cell_label_sorted[i][0][4]
        first_fifth = cell_label_sorted[i][0][5]
        #print("post_labels = "+str(post_labels))
    #add_to_plot_3D(ax, label, colour, post_labels, dim1, dim2, dim3)
    try:
        add_to_plot_3D(ax, label, colour, post_labels, pprs, recoveries, release_probs)
        #add_to_plot_3D(ax, label, colour, post_labels, dim1, dim2, dim3)
        num_added += 1
        print("num_added ="+str(num_added))
        print("added")
    except:
        print('missing entry')
        
    
title_name = "PCA_selectedPhys_100Hz_included_Standard_Acsf_Phys_axes"
x_label = axis_labels[0],
y_label = axis_labels[1],
z_label = axis_labels[2],

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_zlabel(z_label)
#plt.legend()
plt.legend(handles=legend_handles_cluster, loc='lower right')
plt.title(title_name)

#plt.show()


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
            
