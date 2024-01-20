#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:45:10 2023

@author: john
"""
import pickle
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS


#----------------------------------------------------------------------------------
#load in saved arrays
rodent_data_file = open('./processed_input_rodent.p', "rb")
rodent_data = pickle.load(rodent_data_file)

human_data_file = open('./processed_input_human.p', "rb")
human_data = pickle.load(human_data_file)


#print(data)
#num_cols = len(data[0])
#print(data[:,3:])
#params_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
#params_data = [row[3:] for row in data[:]] #model and phys
#params_data = [row[8:] for row in data[:]] #model only

rodent_phys_data = [row[3:8] for row in rodent_data[:]] #phys only w/ 50Hz rec
rodent_hybrid_data = [row[3:] for row in rodent_data[:]] #model and phys
#rodent_model_data = [row[8:] for row in rodent_data[:]] #model only
rodent_model_data = [row[8:13] for row in rodent_data[:]] #model only only first 4
rodent_row_labels = [row[0:3] for row in rodent_data[:]]

human_phys_data = [row[3:8] for row in human_data[:]] #phys only w/ 50Hz rec
human_hybrid_data = [row[3:] for row in human_data[:]] #model and phys
#human_model_data = [row[8:] for row in human_data[:]] #model only
human_model_data = [row[8:13] for row in human_data[:]] #model only only first 4
human_row_labels = [row[0:3] for row in human_data[:]]

#model_data = [row[8:12] for row in data[:]] #only mu kernel data
model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "sigma_baseline", "sigma_amp1", "sigma_amp2", "sigma_amp3", "sigma_scale"]
#model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3"]

phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

#print("printing model data[0]")
#print(model_data[0])

#row_labels = [row[0:3] for row in data[:]]
#params_arr = np.array(params_data)

#--------------------------------------------------------------------------

#merge human and rodent data to fit scaling/PCA
joint_model_data = np.append(rodent_model_data, human_model_data, axis=0)
print("joint_model_data")
print(joint_model_data)

pca = PCA(whiten=True)
scaler = StandardScaler()
scaler.fit(joint_model_data)
scaled_arr = scaler.transform(joint_model_data)
print("joint scaled_arr[0]")
print(scaled_arr[0])
pca_fitted = pca.fit(scaled_arr)
print("pca_fitted.explained_variance_")
print(pca_fitted.explained_variance_)
print("joint model pca components")
print(pca_fitted.components_)
joint_PCA_merged_model = pca_fitted.transform(scaled_arr)
#joint_PCA_merged_model = pca.fit_transform(scaled_arr)
print("joint pca_fitted[0]")
print(joint_PCA_merged_model[0])

#Slice the fitted array
rodent_model_PCA = joint_PCA_merged_model[0:len(rodent_model_data), :] 
human_model_PCA = joint_PCA_merged_model[len(rodent_model_data):, :] 

print("rodent_PCA:")
print(rodent_model_PCA)

print("human_PCA:")
print(human_model_PCA)

#get rodent PCA projection
#scaled_rodent = scaler.transform(rodent_model_data)
#joint_pca_model_result_rodent = pca_fitted.transform(scaled_rodent)

#get human PCA projection
#scaled_human = scaler.transform(human_model_data)
#joint_pca_model_result_human = pca_fitted.transform(scaled_human)

#--------------------------------------------------------------------------
def unsupervised_scatter(train_set, map_set=None, min_samples=8):
    """[Function to perform unsupervised learning and return dimensions by cluster]
    
    :param [train_set]: [input dataset for performing clustering], defaults to [DefaultParamVal]
    :param [map_set]: [input dataset to map clusters to, normally this is the same as 
                       train_set but can be changed to show clustering in an alternate space],
                        defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    #set map_set to train set if none provided, ie. project to same space as the clustering
    if map_set is None:
        map_set=train_set
    
    
    #test logistic regression kernel means
    #clf = LogisticRegression(max_iter=10000)
    cluster_alg = OPTICS(min_samples=min_samples, metric="sqeuclidean")
    cluster_alg.fit(train_set) 
    cluster_predictions = cluster_alg.labels_
    print("PRINTING PREDICTIONS PRINTING PREDICTIONS PRINTING PREDICTIONS PRINTING PREDICTIONS")
    print(cluster_predictions)
    
    prediction_rows = [[] for i in range(0, len(np.unique(cluster_predictions)))]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(cluster_predictions)):
        prediction_rows[cluster_predictions[i]].append(map_set[i][:])
    for i in range(0, len(prediction_rows)):
        #print("prediction "+str(i)+" length = "+str(len(prediction_rows[i])))
        #if len(prediction_rows[i]) > 1: #commented out for testing
        prediction_rows[i] = np.asarray(prediction_rows[i])
            #print(prediction_rows[i])
        """
        else:
            print("conversion failed for prediction "+str(i))
        """
    #return np.asarray(prediction_rows)
    return (prediction_rows, cluster_alg, cluster_predictions)

#--------------------------------------------------------------------------

def surrogate_centroids(input_data, labels, id_index=2):
    """
    pca = PCA(whiten=True)
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_arr = scaler.transform(input_data)
    pca_result = pca.fit_transform(scaled_arr)
    
    clustering = cluster_alg.fit(pca_result)
    labels = clustering.labels_
    """

    #num_clusters = cluster_alg.n_clusters
    num_clusters = len(np.unique(labels))
    cluster_rows = [[] for i in range(0, num_clusters)]
    #create list of lists corresponding to all synapse fit rows by cluster
    
    for i in range(0, len(labels)):
        print("current label:")
        print(labels[i])
        cluster_rows[labels[i]].append(input_data[i][:])

    cluster_means = [[] for i in range(0, num_clusters)]
    for i in range(0, num_clusters):
        cluster_rows[i] = np.asarray(cluster_rows[i])
        print("printing cluster "+str(i))
        print(cluster_rows[i])
        #the below should return a 1d list of feature means for each cluster 
        cluster_means[i] = [np.mean(cluster_rows[i][:, j]) for j in range(0, len(input_data[0]))]
    return (np.asarray(cluster_means), labels)

#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------
#plot empty axes as a figure element

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#x.spines['left'].set_visible(False)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#plt.scatter([1,1],[1,1], alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("J-PC1")
plt.ylabel("J-PC2")
x = [i for i in range(-30, 1000)]
kernels_over_time = []
f.set_size_inches((0.7874, 0.7874))
f.set_dpi(1200)
f.tight_layout()
plt.xlim(-2.3, 2.3)
plt.ylim(-1.5, 2.3)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig3/J-PC_axes_empty.svg"
plt.savefig(save_name, transparent=True)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#x.spines['left'].set_visible(False)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
#plt.scatter([1,1],[1,1], alpha=0.5)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("R-PC1")
plt.ylabel("R-PC2")
x = [i for i in range(-30, 1000)]
kernels_over_time = []
f.set_size_inches((0.7874, 0.7874))
f.set_dpi(1200)
f.tight_layout()
plt.xlim(-2.3, 2.3)
plt.ylim(-1.5, 2.3)
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig3/R-PC_axes_empty.svg"
plt.savefig(save_name, transparent=True)


#--------------------------------------------------------------------------
#do I retain the original clustering PCs and then map labels to new space
#or cluster in new space?
#try both
#print("rodent_model_data")
#print(rodent_model_data)


pca = PCA(whiten=True)
scaler = StandardScaler()
scaler.fit(rodent_model_data)
scaled_arr = scaler.transform(rodent_model_data)
rodent_modelPCA = pca.fit_transform(scaled_arr)
#print("scaled_arr")
#print(scaled_arr)


#rodent_pca_fitted = pca.fit(scaled_arr)
within_pca_model_result_rodent = pca.fit_transform(scaled_arr)
print("within rodent model pca components")
print(pca.components_)
#print("within_pca_model_result_rodent")
#print(within_pca_model_result_rodent)

#project within species PCA cluster labels
rodent_cluster_predictions, rodent_cluster_alg_fitted, rodent_cluster_labels  = unsupervised_scatter(rodent_modelPCA, rodent_model_PCA, min_samples=8) 

#define colours for plotting
red='xkcd:blood red' #e50000
orange='xkcd:pumpkin'
green='xkcd:apple green'
blue='xkcd:cobalt'
purple='xkcd:barney purple'
brown = 'xkcd:brown'
pink='xkcd:hot pink'
taupe='xkcd:taupe'
colour_set = [red, orange, green, blue, purple, brown]

f = plt.figure()
#plt.title("clusters, model PCA 1.3mM")
#plt.title("Rodent Clusters joint PCs")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
x = [i for i in range(-30, 1000)]
kernels_over_time = []
for i in range(len(rodent_cluster_predictions)-1, -1, -1):
    print("plotting cluster #"+str(i))
    print("colour is "+colour_set[i])
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    #label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    label = ""+str(i)
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(rodent_cluster_predictions[i][:, 0], rodent_cluster_predictions[i][:, 1], label=label, color=colour_set[i], alpha=0.7)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.legend()
f.set_size_inches((5.5, 4.3))
f.set_dpi(1200)
f.tight_layout()
plt.xlim(-2.0, 2.3)
plt.ylim(-1.1, 2.3)
plt.xlabel("J-PC1")
plt.ylabel("J-PC2")
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig3/Rodent_jointPCA_Model_cluster_scatter.svg"
plt.savefig(save_name, transparent=True)


pca = PCA(whiten=True)
scaler = StandardScaler()
scaler.fit(human_model_data)
scaled_arr = scaler.transform(human_model_data)
human_pca_fitted = pca.fit_transform(scaled_arr)
human_modelPCA = pca.fit_transform(scaled_arr)
print("within human model pca components")
print(pca.components_)
#print("within_pca_model_result_human")
#print(within_pca_model_result_human)

#project within species PCA cluster labels
human_cluster_predictions, human_cluster_alg_fitted, human_cluster_labels = unsupervised_scatter(human_modelPCA, human_model_PCA, min_samples=10) 
#human_colour_set = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',]
#human_colour_set = ['tab:green', 'tab:orange', 'tab:red', 'tab:blue',  'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue',]
human_colour_set = ['tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'xkcd:sky blue', 'tab:blue' ,'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue',]

#------------------------------------------------------------------------------

surrogate_means, cluster_labels_surr = surrogate_centroids(human_model_data, human_cluster_labels)
mu_kernel_taus = [15, 200, 300] 

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#plt.title("Kernel of SRP OPTICS mean values with baseline") "Kernels by Cluster Human"
plt.title("Kernels by Cluster Human")
x = [i for i in range(-30, 1000)]
kernels_over_time = []
for i in range(0, len(surrogate_means)):
    label = "cluster "+str(i) #+": "+str(syn_by_cluster_surr[i])+" synapses, " #+str(runs_by_cluster_surr[i]) +" runs"
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    title_name = "rodent cluster "+str(i)
    plt.plot(kernel_x, kernel_y, label=label, color=human_colour_set[i]) #match colour to bar plots
    
    #plt.plot(kernel_x, kernel_y)
plt.legend()
f.set_size_inches((3.4, 3.4))
f.set_dpi(1200)
f.tight_layout()
#plt.show()

#------------------------------------------------------------------------------

#rodent: colour_set = [red, orange, green, blue, purple, brown]
f= plt.figure()
#plt.title("clusters, model PCA 1.3mM")
#plt.title("Human Clusters joint PCs")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)
#ax.get_xaxis().set_visible(False)
#ax.get_yaxis().set_visible(False)
x = [i for i in range(-30, 1000)]
kernels_over_time = []
for i in range(len(human_cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    #label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    label = ""+str(i)
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(human_cluster_predictions[i][:, 0], human_cluster_predictions[i][:, 1], color=human_colour_set[i], label=label, alpha=0.7)
    """
    else:
        print("exception on prediction i= "+str(i))
        #plt.plot(kernel_x, kernel_y)
    """
#plt.xlabel("PC1")
#plt.ylabel("PC2")
#plt.legend()
f.set_size_inches((5.5, 4.3))
f.set_dpi(1200)
f.tight_layout()
plt.xlim(-2.0, 2.3)
plt.ylim(-1.1, 2.3)
plt.xlabel("J-PC1")
plt.ylabel("J-PC2")
#plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./fig3/Human_jointPCA_Model_cluster_scatter.svg"
plt.savefig(save_name, transparent=True)


#plot kernel overlays
#use this as a test to verify kernel/cluster match is the same when working with the new PCs

surrogate_means, cluster_labels_surr = surrogate_centroids(rodent_model_data, rodent_cluster_labels)
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
colour_set = [red, orange, green, blue, purple, brown]

single_kernel_dir = "./Figures/single_kernels/"
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

#plt.title("Kernel of SRP OPTICS mean values with baseline") "Kernels by Cluster Human"
plt.title("Kernels by Cluster Rodent")
x = [i for i in range(-30, 1000)]
kernels_over_time = []
for i in range(0, len(surrogate_means)):
    label = "cluster "+str(i) #+": "+str(syn_by_cluster_surr[i])+" synapses, " #+str(runs_by_cluster_surr[i]) +" runs"
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    file_name = single_kernel_dir + "rodent_cluster_"+str(i)
    title_name = "rodent cluster "+str(i)
    plt.plot(kernel_x, kernel_y, label=label, color=colour_set[i]) #match colour to bar plots
    
    #plt.plot(kernel_x, kernel_y)
plt.legend()
f.set_size_inches((3.4, 3.4))
f.set_dpi(1200)
f.tight_layout()
#plt.show()
#save_name = "./Figures/fig3/Unsupervised_kernel_plot.svg"
#plt.savefig(save_name)


#--------------------------------------------------------------------------

"""
#project cross-species PCA cluster labels
rodent_joint_cluster_predictions = unsupervised_scatter(joint_pca_model_result_rodent, joint_pca_model_result_rodent) 

f= plt.figure()
#plt.title("clusters, model PCA 1.3mM")
plt.title("Rodent Joint PCA Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(rodent_joint_cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    #label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    label = ""+str(i)
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(rodent_joint_cluster_predictions[i][:, 0], rodent_joint_cluster_predictions[i][:, 1], label=label)

    #else:
    #    print("exception on prediction i= "+str(i))
    #    #plt.plot(kernel_x, kernel_y)

plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.legend()
f.set_size_inches((2.90, 2.40))
f.set_dpi(1200)
f.tight_layout()
plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./Figures/fig4/Model_jointPCA_rodent_cluster_scatter.svg"
plt.savefig(save_name)

human_joint_cluster_predictions = unsupervised_scatter(joint_pca_model_result_human, joint_pca_model_result_human) 

f= plt.figure()
#plt.title("clusters, model PCA 1.3mM")
plt.title("Human Joint PCA Clusters")
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(len(human_joint_cluster_predictions)-1, -1, -1):
    #print("printing prediction row")
    #print(prediction_rows_post_sct[i])
    #label = ""+str(i)+": "+str(len(cluster_predictions[i][:, 0]))+" synapses" 
    label = ""+str(i)
    #if isinstance(prediction_rows_post_sct[i], np.ndarray):
    plt.scatter(human_joint_cluster_predictions[i][:, 0], human_joint_cluster_predictions[i][:, 1], label=label)

    #else:
    #    print("exception on prediction i= "+str(i))
    #    #plt.plot(kernel_x, kernel_y)

plt.xlabel("PC1")
plt.ylabel("PC2")
#plt.legend()
f.set_size_inches((2.90, 2.40))
f.set_dpi(1200)
f.tight_layout()
plt.legend(bbox_to_anchor=(1.0, 1.0))
save_name = "./Figures/fig4/Model_jointPCA_human_cluster_scatter.svg"
plt.savefig(save_name)

"""