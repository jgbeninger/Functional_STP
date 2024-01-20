#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 14:15:54 2023

@author: john
"""
import numpy as np
from srplasticity.srp import (
    ExpSRP,
    DetSRP,
    ExponentialKernel,
    _convolve_spiketrain_with_kernel,
    get_stimvec)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS 
from sklearn import metrics
import copy

def _default_parameter_bounds(mu_taus, sigma_taus):
    """ returns default parameter boundaries for the SRP fitting procedure """
    return [
        (-4.6, 6),  # mu baseline
        #*[(-10* tau, 10* tau) for tau in mu_taus],  # mu amps, originall7 *-10, *10
        *[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        #*[(-150, 150), (-750, 750), (-1000, 1001), (-3000, 3000)],
        (-6, 6),  # sigma baseline
        *[(-10 * tau, 10 * tau) for tau in sigma_taus],  # sigma amps
        (0.001, 100),  # sigma scale
    ]

def _default_parameter_bounds_simple():
    """ returns default parameter boundaries for the SRP fitting procedure """
    return [
        (-4.6, 6),  # mu baseline
        *[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        (0,3)   #add range for SD here choosing 0 to 3 to reflect top of SST range but not outliers,
        #this will probably be better captured by the Gaussian modelling
    ]

#uniform generation
#assume 
def gen_uniform_srp(param_bounds, num_fits):
    """
    :param param_bounds: The fitting parameter bounds in the form [baseline, amp1, amp2, amp3, SD]
                        where each entry is a tuple of the form (min, max)
    :param num_fits: The number of models to generate from a uniform distribution between parameter
                    bounds
    :return: list of fitted models each stored as tuples 
    """
    fitted_models = [[] for i in range(0, num_fits)]
    for i in range(0, num_fits):
        for j in range(0, len(param_bounds)):
            param = np.random.default_rng().uniform(low=param_bounds[j][0], high=param_bounds[j][1])
            fitted_models[i].append(param)
    return fitted_models

#multivariate normal generation
#numpy.random.multivariate_normal

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
#-----------------------------------------------------------------------------

def scatter_plot_clustering(cluster_predictions, save_name):
    f= plt.figure()
    #plt.title("clusters, model PCA 1.3mM")
    plt.title("Uniform Generation Clusters")
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

#-----------------------------------------------------------------------------

#function to randomly generate cluster labels

def random_clustering(num_samples, min_num_clusters=2, max_num_clusters=14):
    rng = np.random.default_rng()
    num_clusters = rng.integers(low=min_num_clusters, high=max_num_clusters)
    labels = []
    for i in range(0, num_samples):
        labels.append(rng.integers(low=0, high=num_clusters))
    return np.asarray(labels)

#--------------------------------------------------------------------------

param_bounds = _default_parameter_bounds_simple()
print("printing param bounds")
print(param_bounds)

uniform_fits = gen_uniform_srp(_default_parameter_bounds_simple(), num_fits=86) #make the same number of fits are the 1.3mM case 
print("printing uniform fits")
print(uniform_fits)
#get rodent silhoette coefficients

pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())

scaler.fit(uniform_fits)
scaled_rodent = scaler.transform(uniform_fits)
pca_model_result_rodent = pca.fit_transform(scaled_rodent)
min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes
rodent_silhouette_scores = []
x_vals = []
for min_size in min_cluster_sizes:
    try:
        clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_model_result_rodent) 
        print(clustering.labels_)
        rodent_silhouette_scores.append(metrics.silhouette_score(pca_model_result_rodent, clustering.labels_))
        x_vals.append(min_size-1)
        
        #plot clustering
        scatter_plot_clustering(unsupervised_scatter(clustering, pca_model_result_rodent), "./reviewfigs/RodentUniformClusteringMin"+str(min_size))
    except:
        random_labels = random_clustering(86)
        rodent_silhouette_scores.append(metrics.silhouette_score(pca_model_result_rodent, random_labels))
        x_vals.append(min_size-1)
        print("Uniform clustering failed for min size="+str(min_size))
    
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(x_vals, rodent_silhouette_scores) #remaining kmeans
plt.title("Uniform Generation OPTICS Quality")
plt.xticks(x_vals, min_cluster_sizes)
plt.xlabel("Min Cluster Size")
plt.ylabel("Silhouette Score")
save_name = "./reviewfigs/Rodent_uniform_1.3mM_OPTICS_Silhouette_Coefficients.svg"
f.set_size_inches((3.93, 3.59))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)


peak_vals = []
for i in range(0, 10000):
    uniform_fits = gen_uniform_srp(_default_parameter_bounds_simple(), num_fits=86) #make the same number of fits are the 1.3mM case 
    scaler.fit(uniform_fits)
    scaled_rodent = scaler.transform(uniform_fits)
    pca_model_result_rodent = pca.fit_transform(scaled_rodent)
    peak_val = "undefined"
    for min_size in min_cluster_sizes:
        #try:
            clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_model_result_rodent)
            try:
                silhouette_score = metrics.silhouette_score(pca_model_result_rodent, clustering.labels_)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
            except:
                random_labels = random_clustering(86)
                silhouette_score = metrics.silhouette_score(pca_model_result_rodent, random_labels)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
    """
    except:
        print("no clustering found")
    """
    peak_vals.append(peak_val)

print(peak_vals)
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(peak_vals, bins=100)
plt.title("Peak Silhouette Scores By Run")

save_name = "./reviewfigs/uniform_peak_silhouette_coefficients.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)