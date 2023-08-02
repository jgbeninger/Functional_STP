import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

#note naming scheme became inveted "pre" files are in fact post

#1.3mM
with open("./Figures/dists/modelprebootstrapdists1.3mMdownmlp.p", "rb") as input_file:
    model_post_1_3mM = pickle.load(input_file)
    
"""    
with open("./Figures/dists/modelprebootstrapdists1.3mMdownmlp.p", "rb") as input_file:
    model_pre_1_3mM = pickle.load(input_file)
"""
    
with open("./Figures/dists/physprebootstrapdists1.3mMdownmlp.p", "rb") as input_file:
    phys_post_1_3mM = pickle.load(input_file)
  
"""    
with open("./Figures/dists/physprebootstrapdists1.3mMdownmlp.p", "rb") as input_file:
    phys_pre_1_3mM = pickle.load(input_file)
"""  
    
  
#2mM
with open("./Figures/dists/modelprebootstrapdists2mMdownmlp.p", "rb") as input_file:
    model_post_2mM = pickle.load(input_file)
 
    """
with open("./Figures/dists/modelprebootstrapdists2mMdownmlp.p", "rb") as input_file:
    model_pre_2mM = pickle.load(input_file)
"""
    
with open("./Figures/dists/physprebootstrapdists2mMdownmlp.p", "rb") as input_file:
    phys_post_2mM = pickle.load(input_file)

"""    
with open("./Figures/dists/physprebootstrapdists2mMdownmlp.p", "rb") as input_file:
    phys_pre_2mM = pickle.load(input_file)
"""  
    
#pre lists
"""
model_pre_1_3_accuracies, model_pre_1_3_baseline, model_pre_1_3_differences = model_pre_1_3mM
phys_pre_1_3_accuracies, phys_pre_1_3_baseline, phys_pre_1_3_differences = phys_pre_1_3mM

model_pre_2_accuracies, model_pre_2_baseline, model_pre_2_differences = model_pre_2mM
phys_pre_2_accuracies, phys_pre_2_baseline, phys_pre_2_differences = phys_pre_2mM
"""

#post lists
model_post_1_3_accuracies, model_post_1_3_baseline, model_post_1_3_differences = model_post_1_3mM
phys_post_1_3_accuracies, phys_post_1_3_baseline, phys_post_1_3_differences = phys_post_1_3mM

model_post_2_accuracies, model_post_2_baseline, model_post_2_differences = model_post_2mM
phys_post_2_accuracies, phys_post_2_baseline, phys_post_2_differences = phys_post_2mM


#calc diffs
#model_pre_diffs  = np.asarray(model_pre_1_3_differences) - np.asarray(model_pre_2_differences)
model_post_diffs  =  np.asarray(model_post_1_3_differences) - np.asarray(model_post_2_differences)

#phys_pre_diffs  =  np.asarray(phys_pre_1_3_differences) - np.asarray(phys_pre_2_differences)
phys_post_diffs  =  np.asarray(phys_post_1_3_differences) - np.asarray(phys_post_2_differences)

#plot histogram of differences 
"""
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in model_pre_diffs if entry>0])/len(model_pre_diffs)
frac_nonneg = len([1 for entry in model_pre_diffs if entry>=0])/len(model_pre_diffs)
plt.hist(model_pre_diffs, bins=30)
plt.title("Pre_type model 1.3mM vs 2mM histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/Supplementals/multi_label_pre_bootstrap_histmodel1.3_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

"""

#plot histogram of differences 
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in model_post_diffs if entry>0])/len(model_post_diffs)
frac_nonneg = len([1 for entry in model_post_diffs if entry>=0])/len(model_post_diffs)
plt.hist(model_post_diffs, bins=30)
plt.title("Post_type model 1.3mM vs 2mM histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/Supplementals/multi_label_post_bootstrap_histmodel1.3_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)

"""
#plot histogram of differences 
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in phys_pre_diffs if entry>0])/len(phys_pre_diffs)
frac_nonneg = len([1 for entry in phys_pre_diffs if entry>=0])/len(phys_pre_diffs)
plt.hist(model_pre_diffs, bins=30)
plt.title("Pre_type phys1.3mM vs 2mM histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/Supplementals/multi_label_pre_bootstrap_histphys1.3_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
"""

#plot histogram of differences 
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
frac_positive = len([1 for entry in phys_post_diffs if entry>0])/len(phys_post_diffs)
frac_nonneg = len([1 for entry in phys_post_diffs if entry>=0])/len(phys_post_diffs)
plt.hist(model_post_diffs, bins=30)
plt.title("Post_type phys1.3mM vs 2mM histogram, positive frac: "+str(frac_positive )+" nonneg frac: "+str(frac_nonneg))

save_name = "./Figures/Supplementals/multi_label_post_bootstrap_histphys1.3_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_bootstrap_bar_2mM.svg"
#save_name = "./Figures/fig3/multi_label_post_shuffle_bar_1.3mM.svg"
f.set_size_inches((2.3622, 2.465))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)


#plot boxplot of diffs from baseline
"""
#pre
#x_labels = ["phys\n"+pval_to_str(phys_pval),"model\n"+pval_to_str(model_pval), "baseline"]
x_labels = ["phys","model", "phys","model"]
y_vals  = [phys_pre_1_3_differences,  model_pre_1_3_differences, phys_pre_2_differences, model_pre_2_differences]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Pre Type Adjusted Performance")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy - Baseline")
#plt.ylim([0,1])
save_name = "./Figures/Supplementals/baseline_differences_pre_bootstrap.svg"
#save_name = "./Figures/fig3/multi_label_pre_bootstrap_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
f.set_size_inches(((4.72, 2.465)))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)
"""

#post
#x_labels = ["phys\n"+pval_to_str(phys_pval),"model\n"+pval_to_str(model_pval), "baseline"]
x_labels = ["phys","model", "phys","model"]
y_vals  = [phys_post_1_3_differences,  model_post_1_3_differences, phys_post_2_differences, model_post_2_differences]
x_coordinates = [i for i in range(1, len(y_vals)+1)]
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
plt.boxplot(y_vals, showmeans=True, meanline=True, medianprops=medianprops)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Post Type Baseline Difference")
plt.xticks(x_coordinates, x_labels)
#plt.xlabel("Representation")
plt.ylabel("Accuracy - Baseline")
#plt.ylim([0,1])
save_name = "./Figures/Supplementals/baseline_differences_post_bootstrap.svg"
#save_name = "./Figures/fig3/multi_label_pre_bootstrap_1.3mM.svg"
#save_name = "./Figures/fig3/multi_label_pre_shuffle_1.3mM.svg"
f.set_size_inches(((4.72, 2.465)))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(save_name, Transparent=True)