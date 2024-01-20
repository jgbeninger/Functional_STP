#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:27:47 2023

@author: john
"""
import pickle
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS 
from sklearn import metrics

#----------------------------------------------------------------------------------
#before mass saving
"""
def unsupervised_scatter(cluster_predictions, train_set):
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    #cluster_alg = OPTICS(min_samples=min_samples, metric="sqeuclidean")
    #cluster_alg.fit(train_set) 
    #cluster_predictions = cluster_alg.labels_
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
"""

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

#-----------------------------------------------------------------------------

def scatter_plot_clustering(cluster_predictions, save_name, score=None):
    f= plt.figure()
    #plt.title("clusters, model PCA 1.3mM")
    if score==None:
        plt.title("Multivariate Normal Generation Clusters")
    else:
        plt.title("Multivariate Normal Generation Clusters"+str(score))
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
        plt.scatter(cluster_predictions[i][:, 0], cluster_predictions[i][:, 1], label=label, alpha=0.7) #modified to add transparency
        """
        else:
            print("exception on prediction i= "+str(i))
            #plt.plot(kernel_x, kernel_y)
        """
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    #plt.legend()
    #f.set_size_inches((3.30, 2.40))
    f.set_size_inches((2.56, 2.36)) 
    f.set_dpi(1200)
    f.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1.0))
    #save_name = "./fig2/Model_cluster_scatter.svg"
    plt.savefig(save_name+".svg", transparent=True)
    f.clf()
    plt.close()

#-----------------------------------------------------------------------------

row_info = []
data = []
#pyr_types = ['nr5a1', 'rorb']
pyr_indices = []

#l5et_types = ['sim1', 'fam84b']
l5et_indices = []

data_post_pre = {}
unique_pre = set()
unique_post = set()
unique_pairs = set()

measures_file = open('Measures_ex_1.3mM_Human_Type_Pair_Stim.p', "rb")
measures_dict = pickle.load(measures_file)

num_100s = 0
for i in range(2, len(sys.argv)): #starts at 2 for this case because we use input 1 for save number
    cut_on = '_'
    file_name = sys.argv[i]

    print(file_name)
    pair_id = file_name.split(cut_on)[4] #6 for standard, 8 two step, 9 on windows 
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[2] #4 for standard, 6 two step, 7 on windows 
    pre_type = pre_type.split(cut_on)[0]
    pre_type = pre_type.split("/")[1]
    print(pre_type)
    post_type = file_name.split(cut_on)[3] #5 for standard, 7 two step, 8 on windows 
    post_type = post_type.split(cut_on)[0]
    print(post_type)
    
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
    
    
    #add measures
    try:
        key_50hz = ('ic', 50.0, 0.25)
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        
        if ppr > 4: #1000 for the excessive value, note: this was set to 4 until March 9th when changFdataed for some testing, 4 may still be the best value
            print("skipped ppr = "+str(ppr))
            continue
             
        areas = measures_dict[(pre_type, post_type)][pair_id]['areas_50hz_mean']
        release_prob = measures_dict[(pre_type, post_type)][pair_id]['release_prob_all']
        first_fifth = measures_dict[(pre_type, post_type)][pair_id]['first_fifth_50hz_mean']
        first_second = measures_dict[(pre_type, post_type)][pair_id]['first_second_50hz_mean']
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id][key_50hz ]['recovery']
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
    data.append(new_row)
    if post_type in data_post_pre:
        if pre_type in data_post_pre[post_type]:
            curr_data = data_post_pre[post_type][pre_type]
            data_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(new_row)))
        else:
             data_post_pre[post_type][pre_type] = np.array(new_row)
    else:
        data_post_pre[post_type] = {pre_type: np.array(new_row)}

num_cols = len(data[0])
params_data = [row[8:] for row in data[:]] #model only

phys_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
hybrid_data = [row[3:] for row in data[:]] #model and phys
model_data = [row[8:] for row in data[:]] #model only
model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "SD"]

phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

print("printing model data[0]")
print(model_data[0])

row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)

#--------------------------------------------------------------------------
#function to randomly generate cluster labels

def random_clustering(num_samples, min_num_clusters=2, max_num_clusters=14):
    rng = np.random.default_rng()
    num_clusters = rng.integers(low=min_num_clusters, high=max_num_clusters)
    labels = []
    for i in range(0, num_samples):
        labels.append(rng.integers(low=0, high=num_clusters))
    return np.asarray(labels)

#--------------------------------------------------------------------------

#save information for later re-analysis
num_peak_clusters = []
eigenvalues = []
explained_ratio = []
srp_params= []

pca = PCA(whiten=True)
scaler = StandardScaler()
scaler.fit(model_data)
scaled_arr = scaler.transform(model_data)
pca_result = pca.fit_transform(scaled_arr)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

summed_variance = sum(pca.explained_variance_)
first_element = pca.explained_variance_[0] #note should this be multiplied by total variance? seems fine as is
other_diagonals = (summed_variance-first_element)/4
print(str(first_element) +" "+ str(other_diagonals))

#generate covariance matrix elongated in one dimension by the same amount as the real data
cov_mat = np.zeros((5,5))
#cov_mat[(0,0)] = first_element
cov_mat[(0,0)] = 20
cov_mat[(1,1)] = other_diagonals
cov_mat[(2,2)] = other_diagonals
cov_mat[(3,3)] = other_diagonals
cov_mat[(4,4)] = other_diagonals

PCA_like_synthetic_values = np.random.default_rng().multivariate_normal([0,0,0,0,0], cov_mat, 86)


min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes

#uncomment to generate single run with scatterplots
"""
rodent_silhouette_scores = []
x_vals = []
for min_size in min_cluster_sizes:
    print("About to try")
    try:
        clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(PCA_like_synthetic_values) 
        #print(clustering.labels_)
        rodent_silhouette_scores.append(metrics.silhouette_score(PCA_like_synthetic_values, clustering.labels_))
        #plot clustering
        scatter_plot_clustering(unsupervised_scatter(clustering.labels_, PCA_like_synthetic_values), "./reviewfigs/RodentMultiNormClusteringMin"+str(min_size))
    except:
        random_labels = random_clustering(86)
        #print(PCA_like_synthetic_values)
        rodent_silhouette_scores.append(metrics.silhouette_score(PCA_like_synthetic_values, random_labels))
        #plot clustering
        scatter_plot_clustering(unsupervised_scatter(random_labels, PCA_like_synthetic_values), "./reviewfigs/RodentRandomClusteringMin"+str(min_size))
    x_vals.append(min_size-1) 
    
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(x_vals, rodent_silhouette_scores) #remaining kmeans
plt.title("Multivariate Normal Generation OPTICS Quality")
plt.xticks(x_vals, min_cluster_sizes)
plt.xlabel("Min Cluster Size")
plt.ylabel("Silhouette Score")
save_name = "./reviewfigs/Rodent_multinormal_1.3mM_OPTICS_Silhouette_Coefficients.svg"
f.set_size_inches((3.93, 3.59))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)
"""

#repeat many times to generate estimate of peak value distribution

peak_vals = []
for i in range(0, 1000):
    PCA_like_synthetic_values = np.random.default_rng().multivariate_normal([0,0,0,0,0], cov_mat, 139)
    peak_val = "undefined"
    num_clusters = "undefined"
    best_min = "undefined"
    for min_size in min_cluster_sizes:
        #try:
            clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(PCA_like_synthetic_values)
            try:
                silhouette_score = metrics.silhouette_score(PCA_like_synthetic_values, clustering.labels_)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                    num_clusters = (len(np.unique((clustering.labels_))))
                    best_min = min_size
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
                        num_clusters = (len(np.unique((clustering.labels_))))
                        best_min = min_size
            except:
                random_labels = random_clustering(139)
                silhouette_score = metrics.silhouette_score(PCA_like_synthetic_values, random_labels)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                    num_clusters = 1
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
                        num_clusters = 1
     #optional addition: plot the scatter plot with the best value
    try:
        clustering = OPTICS(min_samples=best_min, metric="sqeuclidean").fit(PCA_like_synthetic_values) 
        silhouette_score = metrics.silhouette_score(PCA_like_synthetic_values, clustering.labels_)
        scatter_plot_clustering(unsupervised_scatter(clustering, PCA_like_synthetic_values), "./reviewfigssmall/human_scatters_multivariate_Gaussian/scatter"+str(sys.argv[1])+"_"+str(i), silhouette_score)

    except:
        print("no OPTICS clustering")

    
    """
    except:
        print("no clustering found")
    """
    peak_vals.append(peak_val)
    num_peak_clusters.append(num_clusters)
    gc.collect()
    print(i)

print(peak_vals)
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(peak_vals, bins=100)
plt.title("Peak Silhouette Scores By Run")

save_name = "./reviewfigssmall/human_multinormal_peak_silhouette_coefficients.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

#pickle results for later analysis
results = [peak_vals, num_peak_clusters]
name = "./reviewfigssmall/human_multivariate_Gaussian_pickles/MultivariateGaussianResults" +sys.argv[1] +".p"
file = open(name, "wb")
pickle.dump(results, file)
        