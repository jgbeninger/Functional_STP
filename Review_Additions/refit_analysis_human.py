#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 11:32:18 2023

@author: john
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

pre_peaks_index = 0
post_peaks_index = 1
post_clustnum_index = 3
pre_clustnum_index = 2



with open("./reviewfigssmall/human_refit_pickles/refitresults0.p", "rb") as input_file:
    results0 = pickle.load(input_file)
    
with open("./reviewfigssmall/human_refit_pickles/refitresults1.p", "rb") as input_file:
    results1 = pickle.load(input_file)
    
with open("./reviewfigssmall/human_refit_pickles/refitresults2.p", "rb") as input_file:
    results2 = pickle.load(input_file)
    
#join post peak vals
post_peak_vals = results0[post_peaks_index]
post_peak_vals.extend(results1[post_peaks_index])
post_peak_vals.extend(results2[post_peaks_index])

#join pre peak vals
pre_peak_vals = results0[pre_peaks_index]
pre_peak_vals.extend(results1[pre_peaks_index])
pre_peak_vals.extend(results2[pre_peaks_index])
    

#calculate p-values
sc_above_rodent_pre = ((np.asarray(pre_peak_vals)>0.6433691320170686).sum())/len(pre_peak_vals) #peak value, slightly (negligably) lower for min=8 
sc_above_human_pre = ((np.asarray(pre_peak_vals)>0.7054327142412522).sum())/len(pre_peak_vals)

sc_above_rodent_post = ((np.asarray(post_peak_vals)>0.6433691320170686).sum())/len(post_peak_vals)
sc_above_human_post = ((np.asarray(post_peak_vals)>0.7054327142412522).sum())/len(post_peak_vals)

print("sc_above_rodent_pre="+str(sc_above_rodent_pre))
print("sc_above_human_pre="+str(sc_above_human_pre))
print("sc_above_rodent_post"+str(sc_above_rodent_post))
print("sc_above_human_post="+str(sc_above_human_post))

#generate histograms 
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(pre_peak_vals, bins=100)
#plt.title("Peak Silhouette Coefficient")
plt.axvline(x=0.6433691320170686, color='#55c666ff', ymax=0.9) #stand in for rodent line (85, 198, 102)
plt.axvline(x=0.7054327142412522, color='#a099c5ff', ymax=0.9) #stand in for human line
f.set_size_inches((2.56, 2.06)) #for big-small plot
f.set_dpi(1200)
f.tight_layout()
plt.savefig("./reviewfigssmall/prefit_uniform_histogram_human.svg", transparent=True)

f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.hist(post_peak_vals, bins=100)
#plt.title("Peak Silhouette Coefficient")
plt.axvline(x=0.6433691320170686, color='#55c666ff', ymax=0.9) #stand in for rodent line (85, 198, 102)
plt.axvline(x=0.7054327142412522, color='#a099c5ff', ymax=0.9) #stand in for human line
f.set_size_inches((2.56, 2.06)) #for big-small plot
f.set_dpi(1200)
f.tight_layout()
plt.savefig("./reviewfigssmall/postfit_uniform_histogram_human.svg", transparent=True)

    
#join post nums
nums_post_clusters = results0[post_clustnum_index]
nums_post_clusters.extend(results1[post_clustnum_index])
nums_post_clusters.extend(results2[post_clustnum_index])

#join pre nums
nums_pre_clusters = results0[pre_clustnum_index]
nums_pre_clusters.extend(results1[pre_clustnum_index])
nums_pre_clusters.extend(results2[pre_clustnum_index])

pre_uniques, pre_counts = np.unique(np.asarray(nums_pre_clusters), return_counts=True)
pre_total_counts = len(nums_pre_clusters)

pre_count_probs = pre_counts/pre_total_counts 

print("printing pre count stats")
print(pre_uniques)
print(pre_count_probs)
print(pre_counts)

#plot pre histogram
#save cluster counts
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.bar(pre_uniques, pre_counts)
plt.title("Number of clusters prefitting With Peak Silhouette Score")
plt.show()


uniques, counts = np.unique(np.asarray(nums_post_clusters), return_counts=True)
total_counts = len(nums_post_clusters)

count_probs = counts/total_counts 

#plot post histogram
#save cluster counts
f =plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.axvline(x=5, color='#55c666ff', ymax=0.9, zorder=0) #rodent cluster num
plt.axvline(x=6, color='#a099c5ff', ymax=0.9, zorder=0) #human cluster num
plt.bar(uniques, counts)
#plt.title("Number of clusters postfitting With Peak Silhouette Score")
f.set_size_inches((3.67, 1.56)) #for big-small plot
f.set_dpi(1200)
f.tight_layout()

plt.savefig("./reviewfigssmall/num_clusters_by_peak_SC_post_fitting_human.svg", transparent=True)
plt.show()


#calculate probability of two values within "margin" of each other, for a given
#first value
def prob_product(uniques, counts, val, margin=0):
    #hash implementation would be faster
    p_xk = "undefined"
    summed_prob_products = 0
    unique_index = "undefined"
    for index, unique in enumerate(uniques):
        if val == unique:
            unique_index = index
            p_xk = counts[unique_index]
            print("px1="+str(val)+"="+str(p_xk))
            break
    if unique_index == "undefined":
        #if val has zero probability product will also be zero
        print("value undefined")
        return 0
    print("value defined")
    for i in range(-1*margin, margin):
        #don't test nonexistant values
        if i + unique_index < 0:
            continue
        if i + unique_index > len(counts)-1:
            continue
        #print("p_xk="+str(p_xk))
        #print("counts[i + unique_index]="+str(counts[i + unique_index]))
        #print("product="+str(p_xk*counts[i + unique_index]))
        summed_prob_products = summed_prob_products + p_xk*counts[i + unique_index]
        #print("summed_prob_products="+str(summed_prob_products))
    return summed_prob_products

#calculate co-occurence probabilities within 1 cluster 
total_shared_counts = 0
for i in range(0, uniques[-1]):
    total_shared_counts = total_shared_counts + prob_product(uniques, count_probs, i, margin=1)
    
print(prob_product(uniques, count_probs, 6, margin=1))
print(total_shared_counts)

    
    