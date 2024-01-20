#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 10:42:06 2023

@author: john
"""
import pickle
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS

#------------------------------------------------------------------------------

def unsupervised_scatter(cluster_alg, train_set):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    #cluster_alg = OPTICS(min_samples=min_samples, metric="sqeuclidean")
    #cluster_alg.fit(train_set) 
    cluster_predictions = cluster_alg.labels_
    prediction_rows = [[] for i in range(0, len(np.unique(cluster_predictions)))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(cluster_predictions)):
        prediction_rows[cluster_predictions[i]].append(train_set[i][:])
    for i in range(0, len(prediction_rows)):
        #print("prediction "+str(i)+" length = "+str(len(prediction_rows[i])))
        if len(prediction_rows[i]) > 1:
            prediction_rows[i] = np.asarray(prediction_rows[i])
            #print(prediction_rows[i])
        else:
            print("conversion failed for prediction "+str(i))
    #return np.asarray(prediction_rows)
    return prediction_rows

#------------------------------------------------------------------------------

def scatter_plot_clustering(cluster_predictions, save_name, score=None):
    f= plt.figure()
    #plt.title("clusters, model PCA 1.3mM")
    if score==None:
        plt.title("Uniform Generation Clusters")
    else:
        plt.title("Uniform Generation Clusters"+str(score))
    ax = plt.subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x = [i for i in range(-30, 1000)]
    kernels_over_time = []
    for i in range(len(cluster_predictions)-1, -1, -1):
        #print("printing prediction row")
        #print(prediction_rows_post_sct[i])
        #label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
        label = ""+str(i)
        #if isinstance(prediction_rows_post_sct[i], np.ndarray):
        plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=label)
        """
        else:
            print("exception on prediction i= "+str(i))
            #plt.plot(kernel_x, kernel_y)
        """
    plt.xlabel("R-PC1")
    plt.ylabel("R-PC2")
    #plt.legend()
    #f.set_size_inches((3.30, 2.40))
    f.set_size_inches((4.125, 3.0)) #for big-small plot
    f.set_dpi(1200)
    f.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1.0))
    #save_name = "./fig2/Model_cluster_scatter.svg"
    plt.savefig(save_name, transparent=True)
    f.clf()
    plt.close()

#------------------------------------------------------------------------------

#function to randomly generate cluster labels

def random_clustering(num_samples, min_num_clusters=2, max_num_clusters=14):
    rng = np.random.default_rng()
    num_clusters = rng.integers(low=min_num_clusters, high=max_num_clusters)
    labels = []
    for i in range(0, num_samples):
        labels.append(rng.integers(low=0, high=num_clusters))
    return np.asarray(labels)

#------------------------------------------------------------------------------
#load in physiology + model data 
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

#------------------------------------------------------------------------------

#test clustering on the physiology data
#code carried over from refit_uniform.py
pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())
scaler.fit(phys_data)
scaled_rodent = scaler.transform(phys_data)
pca_model_result_rodent = pca.fit_transform(scaled_rodent)

peak_val = "undefined"
num_clusters = "undefined"
best_min = "undefined"
min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes
silhouette_scores = []
cluster_nums = []
for min_size in min_cluster_sizes:
    #try:
        clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_model_result_rodent)
        try:
            silhouette_score = metrics.silhouette_score(pca_model_result_rodent, clustering.labels_)
            if peak_val == "undefined":
                peak_val = silhouette_score
                num_clusters = (len(np.unique((clustering.labels_))))
                best_min = min_size
            else:
                if silhouette_score > peak_val:
                    peak_val = silhouette_score
                    num_clusters = (len(np.unique((clustering.labels_))))
                    best_min = min_size
            silhouette_scores.append(silhouette_score)
            cluster_nums.append(num_clusters)
        except:
            """
            random_labels = random_clustering(86)
            silhouette_score = metrics.silhouette_score(pca_model_result_rodent, random_labels)
            if peak_val == "undefined":
                peak_val = silhouette_score
                num_clusters = 1 #if except is triggered OPTICS only found one cluster
            else:
                if silhouette_score > peak_val:
                    peak_val = silhouette_score
                    num_clusters = 1 #if except is triggered OPTICS only found one cluster
            
            silhouette_scores.append(silhouette_score)
            cluster_nums.append(num_clusters)
            """
            silhouette_scores.append(-0.5) #marker of failed fit
            cluster_nums.append((len(np.unique((clustering.labels_))))) #marker of failed fit
#optional addition: plot the scatter plot with the best value
#try:
clustering = OPTICS(min_samples=best_min, metric="sqeuclidean").fit(pca_model_result_rodent)
silhouette_score = metrics.silhouette_score(pca_model_result_rodent, clustering.labels_)
scatter_plot_clustering(unsupervised_scatter(clustering, pca_model_result_rodent), "./reviewfigs/scatters_phys/scatter", silhouette_score)


#save figures
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(min_cluster_sizes, silhouette_scores)
plt.title("Phys Silhouette Score By min Cluster Size")

save_name = "./reviewfigs/phys_silhouette_scores.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

#save figures
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(min_cluster_sizes, cluster_nums)
plt.title("Phys Cluster Number By min Cluster Size")

save_name = "./reviewfigs/phys_cluster_nums.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
