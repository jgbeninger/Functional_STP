#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 15:45:10 2023

@author: john
"""
import pickle
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS 
from sklearn import metrics


#----------------------------------------------------------------------------------
#load in saved arrays
rodent_data_file = open('./processed_input_rodent_jan24.p', "rb")
rodent_data = pickle.load(rodent_data_file)

human_data_file = open('./processed_input_human_jan24.p', "rb")
human_data = pickle.load(human_data_file)

rodent_phys_data = [row[3:8] for row in rodent_data[:]] #phys only w/ 50Hz rec
rodent_hybrid_data = [row[3:] for row in rodent_data[:]] #model and phys
rodent_model_data = [row[8:12] for row in rodent_data[:]] #model only
rodent_row_labels = [row[0:3] for row in rodent_data[:]]

human_phys_data = [row[3:8] for row in human_data[:]] #phys only w/ 50Hz rec
human_hybrid_data = [row[3:] for row in human_data[:]] #model and phys
human_model_data = [row[8:12] for row in human_data[:]] #model only
human_row_labels = [row[0:3] for row in human_data[:]]

model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "sigma_baseline", "sigma_amp1", "sigma_amp2", "sigma_amp3", "sigma_scale"]

phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

#--------------------------------------------------------------------------


pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())

#--------------------------------------------------------------------------
#get rodent silhoette coefficients

scaler.fit(rodent_model_data)
scaled_rodent = scaler.transform(rodent_model_data)
pca_model_result_rodent = pca.fit_transform(scaled_rodent)
min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes
rodent_silhouette_scores = []
x_vals = []
for min_size in min_cluster_sizes:
    clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_model_result_rodent) 
    rodent_silhouette_scores.append(metrics.silhouette_score(pca_model_result_rodent, clustering.labels_))
    x_vals.append(min_size-1)
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(x_vals, rodent_silhouette_scores) #remaining kmeans
plt.title("Rodent OPTICS Quality")
plt.xticks(x_vals, min_cluster_sizes)
plt.xlabel("Min Cluster Size")
plt.ylabel("Silhouette Score")
save_name = "./fig2/Rodent_1.3mM_OPTICS_Silhouette_Coefficients.svg"
f.set_size_inches((3.93, 3.59))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)


#get human silhoette coefficients
pca = copy.deepcopy(PCA(whiten=True))
scaler = copy.deepcopy(StandardScaler())

scaler.fit(human_model_data)
scaled_human = scaler.transform(human_model_data)
pca_model_result_human = pca.fit_transform(scaled_human)
min_cluster_sizes = [i for i in range(2, 15)] #specify desired list of minimum cluster sizes
human_silhouette_scores = []
x_vals = []
for min_size in min_cluster_sizes:
    try:
        clustering = OPTICS(min_samples=min_size, metric="sqeuclidean").fit(pca_model_result_human) 
        human_silhouette_scores.append(metrics.silhouette_score(pca_model_result_human, clustering.labels_))
        x_vals.append(min_size-1)
    except:
        print("failed fit for min="+str(min_size)) 

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(x_vals, human_silhouette_scores) #remaining kmeans
plt.title("Human OPTICS Quality")
plt.xticks(x_vals, min_cluster_sizes)
plt.xlabel("Min Cluster Size")
plt.ylabel("Silhouette Score")
save_name = "./fig3/Human_1.3mM_OPTICS_Silhouette_Coefficients.svg"
f.set_size_inches((3.9, 2.1))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name)