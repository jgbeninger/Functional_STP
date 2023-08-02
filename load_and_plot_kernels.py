# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 18:55:16 2022

Script to load saved kernel means from pickle file and plot them

@author: jgben
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import pickle

#--------------------------------------------------
#define taus for kernel
mu_kernel_taus = [15, 100, 300]

#define colours xkcd
red='xkcd:blood red' #e50000
orange='xkcd:pumpkin'
green='xkcd:apple green'
blue='xkcd:cobalt'
purple='xkcd:barney purple'
pink='xkcd:dark pink'
lilac='xkcd:lilac'
periwinkle='xkcd:periwinkle'
bright_purple = 'xkcd:bright purple'
moss_green = 'xkcd:moss green'

#define colours
unsupervised_colours = {0:blue, 1:purple, 2:orange, 3:red, 4:green}
unsupersvied_human_colours = {0:lilac, 1:periwinkle, 2:bright_purple, 3:blue, 4:green, 5:moss_green, 6:red, 7:purple }
supervised_colours = {'sst':red, 'sim1':green, 'nr5a1':blue, 'pvalb':purple, 'tlx3':orange, 'ntsr1': pink}


dir_name_fig1 = "./Figures/fig1/"
dir_name_fig3 = "./Figures/fig3/"
dir_name_fig4 = "./Figures/fig4/"
dir_name_fig5 = "./Figures/fig5/"

#--------------------------------------------------

#unpickel kernel means

#pre raw
filename = 'saved_kernels/data_centers_pre_1.3mM.pickle'
infile = open(filename,'rb')
data_centers_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/raw_unique_pre_1.3mM.pickle'
infile = open(filename,'rb')
raw_pre_types = pickle.load(infile)
infile.close()


filename = 'saved_kernels/group_rows_pre_1.3mM.pickle'
infile = open(filename,'rb')
group_rows_pre = pickle.load(infile)
infile.close()

#post raw
filename = 'saved_kernels/data_centers_post_1.3mM.pickle'
infile = open(filename,'rb')
data_centers_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/raw_unique_post_1.3mM.pickle'
infile = open(filename,'rb')
raw_post_types = pickle.load(infile)
infile.close()

filename = 'saved_kernels/group_rows_post_1.3mM.pickle'
infile = open(filename,'rb')
group_rows_post= pickle.load(infile)
infile.close()


#pre supervised
filename = 'saved_kernels/clf_means_pre_1.3mM.pickle'
infile = open(filename,'rb')
clf_means_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/prediction_rows_pre_1.3mM.pickle'
infile = open(filename,'rb')
prediction_rows_pre = pickle.load(infile)
infile.close()

filename = 'saved_kernels/unique_pre_1.3mM.pickle'
infile = open(filename,'rb')
unique_pre = pickle.load(infile)
infile.close()


#post supervised
filename = 'saved_kernels/clf_means_post_1.3mM.pickle'
infile= open(filename,'rb')
clf_means_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/prediction_rows_post_1.3mM.pickle'
infile = open(filename,'rb')
prediction_rows_post = pickle.load(infile)
infile.close()

filename = 'saved_kernels/unique_post_1.3mM.pickle'
infile = open(filename,'rb')
unique_post = pickle.load(infile)
infile.close()

#rodent unsupervised
filename = 'saved_kernels/OPTICS_means_post_1.3mM.pickle'
infile = open(filename,'rb')
surrogate_means =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_syn_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
syn_by_cluster_surr =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_runs_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
runs_by_cluster_surr =  pickle.load(infile)
infile.close()


#rodent unsupervised
filename = 'saved_kernels/OPTICS_means_post_2mM.pickle'
infile = open(filename,'rb')
surrogate_means_2mM =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_syn_by_cluster_surr_2mM.pickle'
infile = open(filename,'rb')
syn_by_cluster_surr_2mM =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICS_runs_by_cluster_surr_2mM.pickle'
infile = open(filename,'rb')
runs_by_cluster_surr_2mM =  pickle.load(infile)
infile.close()

#human unsupervised
filename = 'saved_kernels/OPTICSHuman_means_post_1.3mM.pickle'
infile = open(filename,'rb')
human_surrogate_means =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICSHuman_syn_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
human_syn_by_cluster_surr =  pickle.load(infile)
infile.close()

filename = 'saved_kernels/OPTICSHuman_runs_by_cluster_surr_1.3mM.pickle'
infile = open(filename,'rb')
human_runs_by_cluster_surr =  pickle.load(infile)
infile.close()

#--------------------------------------------------
#functions

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
    y_vals[0:2000] = mu_baseline
    
    return (x_vals, y_vals)


#--------------------------------------------------
#human plot
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP OPTICS mean values with baseline for human synapses")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(human_surrogate_means)):
    colour = unsupersvied_human_colours[i]
    #label = "cluster "+str(i)+": "+str(human_syn_by_cluster_surr[i])+" synapses, "+str(human_runs_by_cluster_surr[i]) +" runs"
    label = "cluster "+str(i)
    #label = "cluster "+str(i)
    mu_amps = [human_surrogate_means[i,1], human_surrogate_means[i, 2], human_surrogate_means[i,3]] #add 1 to all values
    mu_baseline = human_surrogate_means[i,0] 
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    plt.plot(kernel_x, kernel_y, label=label, color=colour)
    #plt.plot(kernel_x, kernel_y)
plt.legend()
save_name = dir_name_fig5+"humanOPTICS_post_kernels.svg"
plt.savefig(save_name)
#plt.show()  

#--------------------------------------------------

#raw plots

#pre type
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values by pre_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers pre")
print(data_centers_post)
for i in range(0, len(data_centers_pre)):
    if len(group_rows_pre[i]) > 4:
        label = raw_pre_types[i]
        colour = supervised_colours[raw_pre_types[i]]
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_pre[i,1], data_centers_pre[i, 2], data_centers_pre[i,3]] #add 1 to all values
        mu_baseline = data_centers_pre[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label, color = colour)
        #plt.plot(kernel_x, kernel_y)
        """
        else:
            print(unique_post[i])
        """
plt.legend()
save_name = dir_name_fig1+"raw_pre_kernels.svg"
plt.savefig(save_name)
#plt.show()

#post
#post type
mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma
plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("Kernel of SRP mean values by Post_type")
x = [i for i in range(-30,1000)]
kernels_over_time = []
print("printing data centers post")
print(data_centers_post)
for i in range(0, len(data_centers_post)):
    if len(group_rows_post[i]) > 4:
        label = raw_post_types[i]
        colour = supervised_colours[raw_post_types[i]]
        #if len(prediction_rows_post[i]) > 4:
        #print("selected: "+str(unique_post[i]))
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [data_centers_post[i,1], data_centers_post[i, 2], data_centers_post[i,3]] #add 1 to all values
        mu_baseline = data_centers_post[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=label, color=colour)
        #plt.plot(kernel_x, kernel_y)
        """
        else:
            print(unique_post[i])
        """
plt.legend()
save_name = dir_name_fig1+"raw_post_kernels.svg"
plt.savefig(save_name)
#plt.show()


#--------------------------------------------------

#supervised plots
#pre type
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("SVM Predicted Pre Type Kernels")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_pre)):
    if len(prediction_rows_pre[i]) > 1: #testing if at least two points get this classification
        print("selected: "+str(unique_pre[i]))
        colour = supervised_colours[unique_pre[i]]
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [clf_means_pre[i,1], clf_means_pre[i, 2], clf_means_pre[i,3]] #add 1 to all values
        mu_baseline = clf_means_pre[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=unique_pre[i], color=colour)
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_pre[i])
plt.legend()
save_name = dir_name_fig3+"supervised_pre_kernels.svg"
f.set_size_inches((3.11, 2.74))
f.set_dpi(600)
f.tight_layout()
plt.savefig(save_name)
#plt.show()

#post type
f = plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("SVM Predicted Post Type Kernels")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(clf_means_post)):
    if len(prediction_rows_post[i]) > 1: #testing if at least two points get this classification
        print("selected: "+str(unique_post[i]))
        colour = supervised_colours[unique_post[i]]
        #print(prediction_rows[i])
        #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
        mu_amps = [clf_means_post[i,1], clf_means_post[i, 2], clf_means_post[i,3]] #add 1 to all values
        mu_baseline = clf_means_post[i,0] 
        kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
        #kernels_over_time.append(kernel)
        plt.plot(kernel_x, kernel_y, label=unique_post[i], color=colour)
        #plt.plot(kernel_x, kernel_y)
    else:
        print(unique_post[i])
plt.legend()
save_name = dir_name_fig3+"supervised_post_kernels.svg"
f.set_size_inches((3.11, 2.74))
f.set_dpi(600)
f.tight_layout()
plt.savefig(save_name)
#plt.show()


f=plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("SRP OPTICS mean values 1.3mM")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(surrogate_means)):
    colour = unsupervised_colours[i] 
    #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
    label = "cluster "+str(i)
    mu_amps = [surrogate_means[i,1], surrogate_means[i, 2], surrogate_means[i,3]] #add 1 to all values
    mu_baseline = surrogate_means[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    plt.plot(kernel_x, kernel_y, label=label, color=colour)
    #plt.plot(kernel_x, kernel_y)
plt.legend()
save_name = dir_name_fig4+"OPTICS_kernels.svg"
f.set_size_inches((4.25, 2.54))
f.set_dpi(600)
f.tight_layout()
plt.savefig(save_name)
#plt.show()  

f=plt.figure()
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.title("SRP OPTICS mean values 2mM")
x = [i for i in range(-30,1000)]
kernels_over_time = []
for i in range(0, len(surrogate_means_2mM)):
    colour = unsupervised_colours[i] 
    #label = "cluster "+str(i)+": "+str(syn_by_cluster_surr[i])+" synapses, "+str(runs_by_cluster_surr[i]) +" runs"
    label = "cluster "+str(i)
    mu_amps = [surrogate_means_2mM[i,1], surrogate_means_2mM[i, 2], surrogate_means_2mM[i,3]] #add 1 to all values
    mu_baseline = surrogate_means_2mM[i,0] 
    kernel_x, kernel_y = gen_kernel(mu_amps, mu_kernel_taus, 1000, mu_baseline=mu_baseline, dt=1)
    #kernels_over_time.append(kernel)
    print("cluster "+str(i))
    print("mu_amps:")
    print(mu_amps)
    print("mu_baseline: "+str(mu_baseline))
    plt.plot(kernel_x, kernel_y, label=label, color=colour)
    #plt.plot(kernel_x, kernel_y)
plt.legend()
save_name = dir_name_fig4+"OPTICS_kernels_2mM.svg"
f.set_size_inches((4.25, 2.54))
f.set_dpi(600)
f.tight_layout()
plt.savefig(save_name)
