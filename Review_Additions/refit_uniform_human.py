#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:56:05 2023

@author: john
"""
import copy
import sys
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler
import mass_fit_srp2 as srp_fitting
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.cluster import OPTICS 
from srplasticity.srp import ExpSRP
import gc
import pickle

#------------------------------------------------------------------------------

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
    return np.asarray(fitted_models)

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
    f.set_size_inches((2.56, 2.36)) #for big-small plot
    f.set_dpi(1200)
    f.tight_layout()
    #plt.legend(bbox_to_anchor=(1.0, 1.0))
    #save_name = "./fig2/Model_cluster_scatter.svg"
    plt.savefig(save_name+".svg", transparent=True)
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

def _default_parameter_bounds_simple():
    """ returns default parameter boundaries for the SRP fitting procedure"""
    return [
        (-4.6, 6),  # mu baseline
        #actual values
        #*[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        #faciliation test values
        *[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        (0, 3) #testing facilitation shift near deterministic
        #previous value below
        #(0, 1.5)   #add range for SD here choosing 0 to 3 to reflect top of SST range but not outliers,
        #this will probably be better captured by the Gaussian modelling
    ]

#------------------------------------------------------------------------------

def _default_parameter_bounds_second():
    """ returns default parameter boundaries for the SRP fitting procedure"""
    return [
        (-4.6, 6),  # mu baseline
        #actual values
        #*[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        #faciliation test values
        *[(-600, 600), (-4000, 4040), (-12000, 12000)], #hand specify mu kernel bounds #stable supervised
        (0, 0.01) #testing facilitation shift near deterministic
        #previous value below
        #(0, 1.5)   #add range for SD here choosing 0 to 3 to reflect top of SST range but not outliers,
        #this will probably be better captured by the Gaussian modelling
    ]



#------------------------------------------------------------------------------

def _convert_fitting_params(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1 : 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps : 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )

#------------------------------------------------------------------------------

#generate random models
mu_taus = [15, 200, 300] # actual values
#mu_taus = [15, 100, 200, 300] #testing facilitaiton shift values
#stimulus ISIs 
stimulus_dicts = {
    "20": [0] + [50] * 7 + [250] + [50] * 3, 
    "50": [0] + [20] * 7 + [250] + [20] * 3,
    "100": [0] + [10] * 7 + [250] + [100] * 3,
    "50-125":  [0] + [20] * 7 + [125] + [20] * 3,
    "50-500":  [0] + [20] * 7 + [500] + [20] * 3,
    "50-1000":  [0] + [20] * 7 + [1000] + [20] * 3,
}
peak_vals_pre = []
peak_vals_post = [] 

num_clusters_pre = []
num_clusters_post = []

eigenvalues_pre = []
eigenvalues_post = []

explained_ratio_pre = []
explained_ratio_post = []

srp_params_pre = []
srp_params_post = []

min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes
for j in range(0, 1000):
    uniform_fits = gen_uniform_srp(_default_parameter_bounds_simple(), num_fits=139)
    srp_params_pre.append(uniform_fits)
    
    #test clustering on the uniform fits
    pca = copy.deepcopy(PCA(whiten=True))
    scaler = copy.deepcopy(StandardScaler())
    scaler.fit(uniform_fits)
    scaled_rodent = scaler.transform(uniform_fits)
    pca_model_result_rodent = pca.fit_transform(scaled_rodent)
    eigenvalues_pre.append(pca.explained_variance_)
    explained_ratio_pre.append(pca.explained_variance_ratio_)
    
    peak_val = "undefined"
    num_clusters = "undefined"
    best_min = "undefined"
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
            except:
                random_labels = random_clustering(139)
                silhouette_score = metrics.silhouette_score(pca_model_result_rodent, random_labels)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                    num_clusters = 1 #if except is triggered OPTICS only found one cluster
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
                        num_clusters = 1 #if except is triggered OPTICS only found one cluster
    #optional addition: plot the scatter plot with the best value
    #try:
    clustering = OPTICS(min_samples=best_min, metric="sqeuclidean").fit(pca_model_result_rodent)
    silhouette_score = metrics.silhouette_score(pca_model_result_rodent, clustering.labels_)
    scatter_plot_clustering(unsupervised_scatter(clustering, pca_model_result_rodent), "./reviewfigssmall/human_scatters_prefit/scatter"+str(j), silhouette_score)
    """
    except:
        print("no OPTICS clustering")
    """
    
    #append metrics
    peak_vals_pre.append(peak_val)
    num_clusters_pre.append(num_clusters)
    
    #in progress
    #------------------------------------------------------------------------------
    
    #generate new responses from model
    all_responses = []
    for i in range(0, 139):
        #sigma params are placeholders ince we use the SD instead
        #exp_srp_inputs = (uniform_fits[i, 0], (uniform_fits[i, 1:4]), mu_taus,  0, (0,0,0), (0,0,0)) #regular
        exp_srp_inputs = (uniform_fits[i, 0], (uniform_fits[i, 1:4]), mu_taus,  0, (0,0,0), (0,0,0)) #faciliation shift testing
        model = ExpSRP(*exp_srp_inputs)
        
        #generate multiple runs with mean_dict and SD
        responses = { protocol:[] for protocol in stimulus_dicts}
        for key, protocol in stimulus_dicts.items():
            #simulate 15 "runs" per protocol
            #get mean responses
            mean_response, sigma_placeholder, _ = model.run_ISIvec(protocol)
            #sample from Gaussian distribution about the mean
            #simulate 15 "runs" per protocol
            protocol_responses = []
            for step in range(0, len(protocol)):
                #step_responses = np.random.normal(mean_response[step], uniform_fits[i, 4], 5) #regular version
                step_responses = np.random.normal(mean_response[step], uniform_fits[i, 4], 5) #faciliation shift testing
                protocol_responses.append(step_responses)
            responses[key] = np.transpose(np.asarray(protocol_responses))
        all_responses.append(responses)
    
    
    #fit model to new responses
    fitted_models = []
    for i in range(0, len(all_responses)):
        #imported functions from fitting script 
        #print("stim dicts:")
        #print(stimulus_dicts)
        #print("all_responses[i]:")
        #print(all_responses[i])
        #srp_params, optimizer_res = srp_fitting.fit_srp_model(stimulus_dicts, all_responses[i], mu_taus, (0, 0, 0)) #regular
        
        #note, this required a change in mass_fit_srp2.py to take more general case of variable number of taus
        srp_params, optimizer_res = srp_fitting.fit_srp_model(stimulus_dicts, all_responses[i], mu_taus, (0, 0, 0), initial_mu=[0.01,0.01, 0.1], bounds=_default_parameter_bounds_simple()) #faciliation shift testing
        main_params = [srp_params[0]]
        main_params.extend(srp_params[1])
        main_params.append(srp_params[3])
        fitted_models.append(main_params)

    #test clustering on the refitted models
    fitted_models = np.asarray(fitted_models)
    srp_params_post.append(fitted_models)
    #print("fitted models:")
    #print(fitted_models)
    pca = copy.deepcopy(PCA(whiten=True))
    scaler = copy.deepcopy(StandardScaler())
    scaler.fit(fitted_models)
    scaled_rodent = scaler.transform(fitted_models)
    pca_model_result_rodent = pca.fit_transform(scaled_rodent)
    eigenvalues_post.append(pca.explained_variance_)
    explained_ratio_post.append(pca.explained_variance_ratio_)
    peak_val = "undefined"
    num_clusters = "undefined"
    best_min = "undefined"
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
            except:
                random_labels = random_clustering(139)
                silhouette_score = metrics.silhouette_score(pca_model_result_rodent, random_labels)
                if peak_val == "undefined":
                    peak_val = silhouette_score
                    num_clusters = 1 #if except is triggered OPTICS only found one cluster
                else:
                    if silhouette_score > peak_val:
                        peak_val = silhouette_score
                        num_clusters = 1 #if except is triggered OPTICS only found one cluster
    
    #optional addition: plot the scatter plot with the best value
    try:
        clustering = OPTICS(min_samples=best_min, metric="sqeuclidean").fit(pca_model_result_rodent) 
        silhouette_score = metrics.silhouette_score(pca_model_result_rodent, clustering.labels_)
        scatter_plot_clustering(unsupervised_scatter(clustering, pca_model_result_rodent), "./reviewfigssmall/human_scatters_postfit/scatter"+str(j), silhouette_score)
    except:
        print("no OPTICS clustering")
    
    #append metrics
    peak_vals_post.append(peak_val)
    print(j)
    num_clusters_post.append(num_clusters)
    gc.collect()

#save figures

f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(peak_vals_pre, bins=30)
plt.title("Peak Silhouette Scores By Run")

save_name = "./reviewfigssmall/human_uniformprefit_peak_silhouette_coefficients2.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(peak_vals_post, bins=30)
plt.title("Peak Silhouette Scores By Run")

save_name = "./reviewfigssmall/human_uniformpostfit_peak_silhouette_coefficients2.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

#save cluster counts
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(num_clusters_pre, bins=15)
plt.title("Number of clusters With Peak Silhouette Score")

save_name = "./reviewfigssmall/human_uniformprefit_cluster_num.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(num_clusters_post, bins=15)
plt.title("Number of clusters With Peak Silhouette Score")

save_name = "./reviewfigssmall/human_uniformpostfit_cluster_num.svg"
#f.set_size_inches((2.3622, 2.465))
#f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

#print eigenvalues
print("pre eigenvalues")
for val in eigenvalues_pre:
    print(val)

print("post eigenvalues")
for val in eigenvalues_post:
    print(val)

print("pre explained variance ratios")
for val in explained_ratio_pre:
    print(val)

print("post explained variance ratios")
for val in explained_ratio_post: 
    print(val)
    
#pickle results for later analysis
results = [peak_vals_pre, peak_vals_post, num_clusters_pre, num_clusters_post, eigenvalues_pre, eigenvalues_post, explained_ratio_pre, explained_ratio_post]
name = "./reviewfigssmall/human_refit_pickles/refitresults" +sys.argv[1] +".p"
file = open(name, "wb")
pickle.dump(results, file)

#pickle model fits
results = [srp_params_pre, srp_params_post]
name = "./reviewfigssmall/human_refit_pickles/model_params" +sys.argv[1] +".p"
file = open(name, "wb")
pickle.dump(results, file)