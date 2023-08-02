#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 11:31:28 2021

@author: john
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import numpy as np
import pickle
import copy
import scipy.stats as stat
import math
import sys

from srplasticity.srp import (
    ExpSRP,
    ExponentialKernel,
    _convolve_spiketrain_with_kernel,
    get_stimvec,
)
from srplasticity.inference import fit_srp_model_gridsearch
#loaded tm to deal with conditional below, clean this up
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel

#----------------------------------------------------------------------------------

#dictionary mapping cre-types to capital
cre_capital = {
    'pvalb':'Pvalb',
    'nr5a1':'Nr5a1',
    'sim1':'Sim1',
    'sst':'Sst',
    'vip':'Vip',
    'ntsr1':'Ntsr1',
    'tlx3':'Tlx3',
    'fam84b':'Fam84b',
    'rorb':'Rorb'
    }

#----------------------------------------------------------------------------------

def get_model_estimates(model, stimulus_dict):
    """
    :return: Model estimates for training dataset
    """
    estimates = {}
    if isinstance(model, ExpSRP):
        means = {}
        sigmas = {}
        for key, isivec in stimulus_dict.items():
            means[key], sigmas[key], estimates[key] = model.run_ISIvec(
                isivec, ntrials=10000
            )
        return means, sigmas, estimates

    elif isinstance(model, TsodyksMarkramModel):
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)
            model.reset()

        return estimates

    else:
        for key, isivec in stimulus_dict.items():
            estimates[key] = model.run_ISIvec(isivec)

        return estimates

#----------------------------------------------------------------------------------

markersize = 3
capsize = 2
lw = 1
def plot_mufit(axis, target_dict_20, target_dict_50, srp_mean):

    xax = np.arange(12)
    print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))

    #"""  
    axis.errorbar(
        xax,
        np.nanmean(target_dict_20, 0),
        yerr=stat.sem(target_dict_20, 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label="20 Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    
    axis.errorbar(
        xax,
        np.nanmean(target_dict_50, 0),
        yerr=stat.sem(target_dict_50, 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_50, 0) / 2,
        #yerr=np.nanstd(target_dict_50, 0),
        color="black",
        marker="s",
        label="50 Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    

    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    axis.plot(srp_mean["20"], color=color["srp"], ls="dashed", zorder=10)
    axis.plot(srp_mean["50"], color=color["srp"], zorder=10)

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    axis.legend(frameon=False)

#----------------------------------------------------------------------------------

def plot_mufit2(axis, target_dict, srp_mean):

    xax = np.arange(12)
    print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))

    #"""
    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    for key in srp_mean.keys():
        axis.errorbar(
            xax,
            np.nanmean(target_dict[key], 0),
            yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
            #yerr=np.nanstd(target_dict_20, 0) / 2,
            #yerr=np.nanstd(target_dict_20, 0),
            color="black",
            ls="dashed",
            marker="o",
            label=""+str(key)+" Hz",
            capsize=capsize,
            elinewidth=0.7,
            lw=lw,
            markersize=markersize,
        )
        axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)
        

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    #axis.legend(frameon=False)

#----------------------------------------------------------------------------------

def plot_mufit3(axis, target_dict, srp_mean, key):

    xax = np.arange(12)
    #print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))
    #print(target_dict)
    #print(np.nanmean(target_dict[key], 0))
    
    #"""
    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    #for key in srp_mean.keys():
    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        #np.nanmedian(target_dict[key], 0),
        yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label=""+str(key)+" Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)
        

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    #axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    #axis.legend(frameon=False)

#----------------------------------------------------------------------------------

def plot_mufit4(f, ax, ax2, target_dict, srp_mean, key):

    xax = np.arange(12)
    #print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))
    #print(target_dict)
    #print(np.nanmean(target_dict[key], 0))
    
    #"""
    """
    color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}
    #for key in srp_mean.keys():
    axis.errorbar(
        xax,
        np.nanmean(target_dict[key], 0),
        #np.nanmedian(target_dict[key], 0),
        yerr=stat.sem(target_dict[key], 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        #yerr=np.nanstd(target_dict_20, 0),
        color="black",
        ls="dashed",
        marker="o",
        label=""+str(key)+" Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )
    axis.plot(srp_mean[key], color=color["srp"], ls="dashed", zorder=10)
    

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    #axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    #axis.legend(frameon=False)
    """


    x_coordinates = [i for i in range(1, 13)]
    x_coordinates_1 = [i for i in range(1, 9)]
    x_coordinates_2 = [i for i in range(9, 13)]
    #plt.figure()
    #ax = plt.subplot(111)
    #f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
    f.set_size_inches((3.15, 1.97))
    f.set_dpi(600)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_visible(False)
    #plt.bar(x_coordinates, y_vals, yerr=y_errors)
    #plt.plot(x_coordinates, model_vals, label="Model")
    
    print("printing raw_sems")
    print(stat.sem(target_dict[key], 0, nan_policy='omit')[0:8])
    print("printing raw vals")
    print(np.nanmean(target_dict[key],0)[0:8])
    #plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data") #changed to sem
    ax.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=stat.sem(target_dict[key], 0, nan_policy='omit'), label="Data", color="black",
    ls="dashed", capsize=capsize, elinewidth=0.7) 
    ax2.errorbar(x_coordinates, np.nanmean(target_dict[key], 0), yerr=stat.sem(target_dict[key], 0, nan_policy='omit'), label="Data", color="black",
    ls="dashed", capsize=capsize, elinewidth=0.7)#changed to sem
    
    ax.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black") 
    ax2.scatter(x_coordinates, np.nanmean(target_dict[key], 0), color="black") 
    
    ax.plot(x_coordinates, srp_mean[key], label="Model", color="#cc3311")
    ax2.plot(x_coordinates, srp_mean[key], label="Model", color="#cc3311") 
    
    #set limits to divide subplots over delay
    ax.set_xlim(0.5,8.01)
    ax2.set_xlim(8.98,12.5)
    
    #add diagonal lines to delay break
    d = .01 # how big to make the diagonal lines in axes coordinates initially 0.015
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-(d/2),1+(d/2)), (-d,+d), linewidth=3, **kwargs)
    #ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    #ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d+0.01,+d+0.01), (-d,+d), linewidth=3, **kwargs)
    
    
    #plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
    #f.suptitle("Model Fit For "+pre_type +" Pre_type", fontsize =16)
    #ax.set_title("Mean Value of Model and Data by Stimulus for "+post_type+" Post_type")
    #plt.legend()
    #ax2.legend()
    
    """
    plt.xticks(x_coordinates, x_coordinates)
    plt.xlabel("Stimulus #")
    plt.ylabel("Normalized Amplitude")
    """
    
    ax.set_xticks(x_coordinates_1)
    ax2.set_xticks(x_coordinates_2)
    #ax.set_xlabel("Stimulus #") 
    #plt.xlabel("Stimulus #")
    #f.text(0.5, 0.02, 'Stimulus #', ha='center')
    #ax.set_ylabel("Normalized Amplitude") 
    
    #plt.ylim([0,1])
    #save_name = fig_folder+"model_data_fit_pre_"+pre_type+".svg"
    #plt.savefig(save_name)

#----------------------------------------------------------------------------------

pre_type = 'nr5a1'
post_type = 'vip'
chosen_pair = 77886
max_threshold = 0.01

pickle_file = open("Extracted_STP_1.3mM_Rodent.p", "rb")
#pickle_file = open('Extracted_ex_1.3mM_Human_Type_Pair_Stim.p', "rb")
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
recordings = pickle.load(pickle_file)
#print(recordings)
#unneeded_pre_types = ['nr5a1', 'unknown'] 
target_dict = {}
training_stim_dict = {} 
for type_pair in recordings.keys():      
    type1, type2 = type_pair
    """
    if type1 != pre_type:
        continue
    if type2 != post_type:
        continue  
    """
    
    chosen_dict = recordings[type_pair] #should be sst
    #print(chosen_dict)
        
    #target_arr_20 = np.asarray(target_list_20)
    #target_arr_100 = np.asarray(target_list_100)
    #trim these to first 8
    #print(chosen_dict[('ic', 20.0, 0.253)])
    #target_dict = {}
    #training_stim_dict = {} #put this back into use on lines 289 and 290 for version 2
    #print(chosen_dict.keys())
    #print(target_arr_20)
    #stuff below is from version 2
    
    print("About to print keys")
    print(chosen_dict.keys())
    for pair_id in chosen_dict.keys():
        """
        if pair_id != chosen_pair:
            continue
        """
        #print(key)
        #print(chosen_dict['pair_IDs'])
        if pair_id != 'pair_IDs':
            """
            try: 
                target_dict[pair_id]
            except:
                target_dict[pair_id] = { "pair_ID": pair_id}
            """
            #rint(key)
            #print(chosen_dict[key])
            #clamp, freq, delay = key #unpack tuple key for conditions
            #clamp, freq, delay = key
            #if clamp == 'ic': #select only current clamp
                #try:
                #    #np.append(target_dict[str(int(freq))], chosen_dict[key][:, 0:8])
                #determine row wise average for divisor
                
                #modify to work by pairs
            first_spike_list = []
            testing_counter = 0
            for protocol in chosen_dict[pair_id]:
                """
                try: 
                    target_dict[pair_id]
                except:
                    target_dict[pair_id] = { "pair_ID": pair_id}
                """
                testing_counter += 1
                if not isinstance(protocol, int):
                    clamp, freq, delay = protocol
                    print(protocol)
                    #first_spike_list = []
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            #changed to make average excluding rows with any column values below 1E-9
                            """
                            safe_row = True 
                            for n in range(0, 8):
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                            """
                            divisor = chosen_dict[pair_id][protocol][i, 0]
                            #print("divisor = "+str(divisor))
                            #if divisor > 1E-6:
                            if divisor > 1E-9:
                                if divisor < max_threshold:
                                    first_spike_list.append(divisor)
                            else: 
                                first_spike_list.append(1E-9)
                        #if len(first_spike_list) > 0:
            #print(testing_counter)
            print(pair_id)
            #print(protocol)
            print(len(first_spike_list))
            print("printing first_spike_list")
            #remove sort after testing
            #first_spike_list.sort()
            print(first_spike_list)
            if len(first_spike_list) > 0:
                averaged_divisor = sum(first_spike_list)/len(first_spike_list)
                #apply normalisation
            else:
                print("voltage clamp runs only for pair")
            for protocol in chosen_dict[pair_id]:
                #for i in range(0, len(first_spike_list)):
                #divisor = chosen_dict[key][i, 0]
                #print("divisor = "+str(divisor))
                if not isinstance(protocol, int):
                    #print(protocol)
                    clamp, freq, delay = protocol
                    if clamp == 'ic':
                        for i in range(0, len(chosen_dict[pair_id][protocol])):
                            added_row = chosen_dict[pair_id][protocol][i,:]
                            safe = True 
                            for n in range(0, len(added_row)):
                                """
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                                """
                                if chosen_dict[pair_id][protocol][i, n] < 1E-9:
                                    added_row[n] = 1E-9
                                if chosen_dict[pair_id][protocol][i, n] > max_threshold:
                                    safe = False
                            #skip row if values above threshold=
                            if not safe:
                                continue
                            #if divisor > 1E-5:
                            normed_row = added_row/averaged_divisor
                            #print(normed_row)
                            try:
                                target_dict[pair_id][(freq, delay)] = np.vstack((target_dict[pair_id][(freq, delay)], normed_row))
                                #print("try")
                                #print(target_dict)
                                #print("append success")
                            except:
                                #target_dict[str(int(freq))] = chosen_dict[key][:, 0:8]
                                #print(freq)
                                try:
                                    target_dict[pair_id][(freq, delay)] = normed_row
                                except: 
                                    target_dict[pair_id] = {(freq, delay): normed_row}
                                #print(int(freq))
                                try:
                                    training_stim_dict[pair_id][(freq, delay)] = [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3 #should be [0] + ...
                                except:
                                    training_stim_dict[pair_id] = {(freq, delay): [0] + [1000/freq] * 7 + [delay*1000] + [1000/freq]*3}
    

#----------------------------------------------------------------------------------

#pkl1 = "fixed_delay_fits/srp_fit_fixed_delay_nr5a1_pvalb91896.p"
#kl2 = "fixed_delay_fits/srp_fit_fixed_delay_nr5a1_pvalb91893.p"

"""

pkl1 = "mass_fits2/srp_fit_full_nr5a1_sst_89701.p"
pkl2 = "mass_fits2/srp_fit_full_nr5a1_sst_94545.p"

pre_type = 'nr5a1'
post_type = 'sst'

pair_id1= 89701
pair_id2 = 94545
pair_id_list = [pair_id1, pair_id2]

pkl1_file = open(pkl1, "rb")
pkl2_file = open(pkl2, "rb")

params1 = pickle.load(pkl1_file)
params2 = pickle.load(pkl2_file)


print("printing parameters")
print(params1)
print(params2)

srp_mean1, srp_sigma1, srp_est1 = get_model_estimates(ExpSRP(*params1), training_stim_dict[pair_id1])
srp_mean2, srp_sigma2, srp_est2 = get_model_estimates(ExpSRP(*params2), training_stim_dict[pair_id2])
srp_means = {pair_id1: srp_mean1, pair_id2: srp_mean2}
"""
pair_id_list = []
type_list = []
#pre_type = 'sim1'
#post_type = 'sst'
post_cut = pre_type+"_" +post_type +"_"
#post_cut = post_type+"_"
#dir_name  = "multi_plots/"
#dir_name  = "multi_plots5_tau2/"
#dir_name = "mass_plots6_tau2/"
#dir_name = "mass_plots6_2mM/"
#dir_name = dir_name = "mass_plots_two_step_1.3mM/" #last
#dir_name = dir_name = "mass_plots_two_step_v2/"
dir_name = "./fig1/indfits/"
#dir_name = dir_name = "mass_plots_human/"

srp_means = {}
for i in range(1, len(sys.argv)):
    file_name = sys.argv[i]
    print(file_name)
    pair_id = file_name.split('_')[5]
    pre_type= file_name.split('_')[3] 
    pre_type = pre_type.split("/")[1]
    post_type= file_name.split('_')[4]
    type_list.append((pre_type, post_type))
    #pair_id = file_name.split(post_cut, 1)[1]
    print("pre_type: "+pre_type)
    print("post_type: "+post_type)
    print("split1: "+pair_id)
    pair_id = pair_id.split(".p",1)[0]
    print("split2: "+pair_id)
    pair_id = int(pair_id)
    pair_id_list.append(pair_id)
    pkl1_file = open(file_name, "rb")
    params1 = pickle.load(pkl1_file)
    
    mu_baseline, mu_amps, mu_taus, SD, mu_scale = params1
    #set dummy vals for sigma params, should not effect plots
    sigma_baseline=[-1.8]
    sigma_amps=[0.1,0.1,0.1]
    sigma_scale=[4]
    sigma_taus = [15, 100, 300]
    mu_scale=None
    constructed_params = (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )
    srp_mean1, srp_sigma1, srp_est1 = get_model_estimates(ExpSRP(*constructed_params), training_stim_dict[pair_id])
    srp_means[pair_id] = srp_mean1

for i in range(0, len(pair_id_list)):
    pair_id = pair_id_list[i]
    pre_type, post_type = type_list[i]
    keys = target_dict[pair_id].keys()
    protocols = []
    for key in keys:
        print(key)
        if key != 'pair_ID':
            protocols.append(key)
        
    num_rows = int(math.ceil(len(protocols)/3))
   # print(protocols)
    #fig, axs = plt.subplots(num_rows, 3, figsize =(15, 4*num_rows) )
    #print(axs)
    #fig.suptitle(pre_type + " to "+ post_type+": "+str(pair_id), fontsize=16)
    for index in range(0, len(protocols)):
        print("pair_ID: "+str(pair_id))
        x = int(index/3)
        y = index % 3
        print("len(protocols) = "+str(len(protocols)))
        print("x="+str(x)+" y="+str(y)+" key ="+str(protocols[index]))
        num_runs = len(target_dict[pair_id][protocols[index]])
        if target_dict[pair_id][protocols[index]].ndim < 2:
            print("Dimension too low")
            continue
        plottitle = cre_capital[pre_type]+" to "+cre_capital[post_type]+" "+str(protocols[index][0])+"Hz "+str(protocols[index][1])+"s"
        title = "Protocol: " + str(protocols[index][0])+"Hz, "+ str(protocols[index][1])+"s, "+str(num_runs)+" runs"
        savetitle = str(protocols[index][0])+"Hz, "+"_"+str(protocols[index][1])+"s"+"_"+str(num_runs)+" runs"
        #plottitle = str(protocols[index][0])+"Hz, "+"_"+str(protocols[index][1])+"s"
        #don't plot if only one run
        #if len(protocols) > 2:
        f = plt.figure()
        ax = plt.axes()
        f.set_size_inches((3.15, 1.97))
        f.set_dpi(1200) 
        #plot_mufit3(ax, target_dict[pair_id], srp_means[pair_id], protocols[index])
        f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
        try:
            plot_mufit4(f, ax, ax2, target_dict[pair_id], srp_means[pair_id], protocols[index])
            #ax.set_title(title)
            #f.suptitle(pre_type + "_"+ post_type + "_"+plottitle)
            f.suptitle(plottitle)
            #f.tight_layout() 
            f.tight_layout()
            save_name = dir_name + pre_type + "_"+ post_type + "_"+savetitle + "_"+ str(pair_id)+".svg"
            plt.savefig(save_name, transparent=True)
            """
            else:
                plot_mufit3(axs[y], target_dict[pair_id], srp_means[pair_id], protocols[index])
                axs[y].set_title(title)
            """
        except:
            print("failed plot")
            continue
    """
    fig.tight_layout() 
    save_name = dir_name + pre_type + "_"+ post_type + "_" + str(pair_id)
    plt.savefig(save_name)
    """
    #plt.show()
    
#plot_mufit3(axis, target_dict, srp_mean, key):    
"""    
print("printing target array 50")
print(target_dict[pair_id1]["100"])
fig, ax = plt.subplots()
plot_mufit(axs[0,0], target_dict[pair_id1]["20"], target_dict[pair_id1]["50"], srp_mean1) #should be srp_mean
#plt.show()


print("printing target array 50")
print(target_dict[pair_id2]["50"])
#fig, ax = plt.subplots()
plot_mufit(axs[0,1], target_dict[pair_id2]["20"], target_dict[pair_id2]["50"], srp_mean2) #should be srp_mean
plt.show()
"""

"""

fig, axs = plt.subplots(2, 2)
print("plotting all freqs")
fig, ax = plt.subplots()
#plot_mufit2(ax, target_dict[pair_id1], srp_mean1)
plot_mufit2(axs[0,0], target_dict[pair_id1], srp_mean1)
#plt.show()

print("plotting all freqs")
#fig, ax = plt.subplots()
#plot_mufit2(ax, target_dict[pair_id2], srp_mean2)
plot_mufit2(axs[0,1], target_dict[pair_id2], srp_mean2)
plt.show()
"""
#----------------------------------------------------------------------------------

