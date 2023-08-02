#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:18:24 2023

@author: john
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


file_names_pre_phys = ["physprealgcompdists1.3mMfulln1000r1.p",
                       "physprealgcompdists1.3mMfulln3000r2.p",
                       "physprealgcompdists1.3mMfulln3000r3.p"
                       ]

file_names_pre_model = ["modelprealgcompdists1.3mMfulln1000r1.p",
                       "modelprealgcompdists1.3mMfulln3000r2.p",
                       "modelprealgcompdists1.3mMfulln3000r3.p"
                       ]

file_names_post_phys = [#"physpostalgcompdists1.3mMfulln100r1.p",
                        "physpostalgcompdists1.3mMfulln3000r1.p",
                        "physpostalgcompdists1.3mMfulln3000r2.p",
                        "physpostalgcompdists1.3mMfulln3000r3.p"
                        ]
file_names_post_model = [#"modelpostalgcompdists1.3mMfulln100r1.p",
                         "modelpostalgcompdists1.3mMfulln3000r1.p",
                         "modelpostalgcompdists1.3mMfulln3000r2.p",
                         "modelpostalgcompdists1.3mMfulln3000r3.p"]

alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm"]
pre_phys_accs = [[] for i in range(0, len(alg_labels))]
pre_phys_baselines = [[] for i in range(0, len(alg_labels))]
for filename in file_names_pre_phys:
    file = 'Figures/dists3/'+ filename 
    infile = open(file,'rb')
    phys_pre_alg_dists, alg_labels = pickle.load(infile)
    infile.close()
    for i in range(0, len(alg_labels)):
        pre_phys_accs[i] = pre_phys_accs[i] + phys_pre_alg_dists[i][0]
        pre_phys_baselines[i] = pre_phys_baselines[i] + phys_pre_alg_dists[i][1]
    
pre_model_accs = [[] for i in range(0, len(alg_labels))]
pre_model_baselines = [[] for i in range(0, len(alg_labels))]
for filename in file_names_pre_model:
    file = 'Figures/dists3/'+ filename 
    infile = open(file,'rb')
    model_pre_alg_dists, alg_labels = pickle.load(infile)
    infile.close()
    for i in range(0, len(alg_labels)):
        pre_model_accs[i] = pre_model_accs[i] + model_pre_alg_dists[i][0]
        pre_model_baselines[i] = pre_model_baselines[i] + model_pre_alg_dists[i][1]
        
post_model_accs = [[] for i in range(0, len(alg_labels))]
post_model_baselines = [[] for i in range(0, len(alg_labels))]
for filename in file_names_post_model:
    file = 'Figures/dists3/'+ filename 
    infile = open(file,'rb')
    model_post_alg_dists, alg_labels = pickle.load(infile)
    infile.close()
    for i in range(0, len(alg_labels)):
        post_model_accs[i] = post_model_accs[i] + model_post_alg_dists[i][0]
        post_model_baselines[i] = post_model_baselines[i] + model_post_alg_dists[i][1]
        
post_phys_accs = [[] for i in range(0, len(alg_labels))]
post_phys_baselines = [[] for i in range(0, len(alg_labels))]
for filename in file_names_post_phys:
    file = 'Figures/dists3/'+ filename 
    infile = open(file,'rb')
    phys_post_alg_dists, alg_labels = pickle.load(infile)
    infile.close()
    for i in range(0, len(alg_labels)):
        post_phys_accs[i] = post_phys_accs[i] + phys_post_alg_dists[i][0]
        post_phys_baselines[i] = post_phys_baselines[i] + phys_post_alg_dists[i][1]
        
        
#model post
alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm", "baseline" ]
#x_labels = alg_labels
#x_labels = alg_labels.append( "baseline")
print(post_model_accs[0])
print(post_model_baselines[0])
post_model_accs.append(post_model_baselines[0])
y_vals  = [np.asarray(post_model_accs[i]) for i in range(0, len(post_model_accs))]
#y_vals = y_vals.append(np.asarray(post_model_baselines[0]))
#print(y_vals)
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Postsynaptic Subclass Prediction By Algorithm Model")
plt.xticks(x_coordinates, alg_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy")
#plt.ylim([0,1])
#save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
#f.set_size_inches(((2.3622, 2.465)))
#f.set_dpi(1200)
f.tight_layout()
save_name = "./Figures/Supplementals/alg_comp_post_model.svg"
plt.savefig(save_name, transparent=True)
plt.show()


#model post
alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm", "baseline" ]
#x_labels = alg_labels
#x_labels = alg_labels.append( "baseline")
print(post_phys_accs[0])
print(post_phys_baselines[0])
post_phys_accs.append(post_phys_baselines[0])
y_vals  = [np.asarray(post_phys_accs[i]) for i in range(0, len(post_phys_accs))]
#y_vals = y_vals.append(np.asarray(post_model_baselines[0]))
#print(y_vals)
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Postsynaptic Subclass Prediction By Algorithm Phys")
plt.xticks(x_coordinates, alg_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy")
#plt.ylim([0,1])
#save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
#f.set_size_inches(((2.3622, 2.465)))
#f.set_dpi(1200)
f.tight_layout()
save_name = "./Figures/Supplementals/alg_comp_post_phys.svg"
plt.savefig(save_name, transparent=True)
plt.show()

        
#phys pre
alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm", "baseline" ]
#x_labels = alg_labels.append( "baseline")
baseline_accs = np.asarray(pre_phys_accs[0][1])
print(baseline_accs)
pre_phys_accs.append(pre_phys_baselines[0])
y_vals  = [np.asarray(pre_phys_accs[i]) for i in range(0, len(pre_phys_accs))]
print(y_vals)
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Presynaptic Subclass Prediction By Algorithm Phys")
plt.xticks(x_coordinates, alg_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy")
#plt.ylim([0,1])
#save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
#f.set_size_inches(((2.3622, 2.465)))
#f.set_dpi(1200)
f.tight_layout()
save_name = "./Figures/Supplementals/alg_comp_pre_phys.svg"
plt.savefig(save_name, transparent=True)
plt.show()

#model pre
alg_labels = ["gb", "lr",  "adb", "mlp", "rf", "svm", "baseline"]
#x_labels = alg_labels.append( "baseline")
baseline_accs = np.asarray(pre_model_accs[0][1])
print(baseline_accs)
pre_model_accs.append(pre_model_baselines[0])
y_vals  = [np.asarray(pre_model_accs[i]) for i in range(0, len(pre_model_accs))]
print(y_vals)
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Presynaptic Subclass Prediction By Algorithm Model")
plt.xticks(x_coordinates, alg_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy")
#plt.ylim([0,1])
#save_name = "./Figures/fig3/multi_label_pre_bootstrapv2_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
#f.set_size_inches(((2.3622, 2.465)))
#f.set_dpi(1200)
f.tight_layout()
save_name = "./Figures/Supplementals/alg_comp_pre_model.svg"
plt.savefig(save_name, transparent=True)
plt.show()
