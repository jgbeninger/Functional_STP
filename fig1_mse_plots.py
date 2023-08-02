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
from scipy.stats import ttest_ind

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
#figure folder
fig_folder = "./fig1/"

#----------------------------------------------------------------------------------
#define colour schems
#unsupervised_colours = {0:, 1:, 2:, 3:, 4:}
  
#dictionary to capitalize cre-types  
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
    #print("testing arrays from fit: nanmean, nanstd for 20hz")
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
    #print("testing arrays from fit: nanmean, nanstd for 20hz")
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

pre_type = 'nr5a1'
post_type = 'vip'
chosen_pair = 77886
max_threshold = 0.01

pickle_file = open("Extracted_STP_1.3mM_Rodent.p", "rb")
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
recordings = pickle.load(pickle_file)
#print(recordings)
#unneeded_pre_types = ['nr5a1', 'unknown'] 
target_dict = {}
training_stim_dict = {} 

pre_types = []
post_types = []
for type_pair in recordings.keys():      
    type1, type2 = type_pair
    pre_types.append(type1)
    post_types.append(type2)
unique_pres = np.unique(np.asarray(pre_types))
unique_posts = np.unique(np.asarray(post_types))
pre_mses = {}
pre_fits = {}
for pre in unique_pres:
    #pre_mses[pre] = [[] for i in range(0, 12)] 
    pre_mses[pre] = []
    pre_fits[pre] = {}

post_mses = {}
post_fits = {}
for post in unique_posts:
    #post_mses[post] = [[] for i in range(0, 12)]
    post_mses[post] = []
    post_fits[post] = {}
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
    
    #print("About to print keys")
    #print(chosen_dict.keys())
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
            #print(pair_id)
            #print(protocol)
            #print(len(first_spike_list))
            #print("printing first_spike_list")
            #remove sort after testing
            #first_spike_list.sort()
            #print(first_spike_list)
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
#pre_type = 'sim1'
#post_type = 'sst'
post_cut = pre_type+"_" +post_type +"_"
#post_cut = post_type+"_"
#dir_name  = "multi_plots/"
#dir_name  = "multi_plots5_tau2/"
#dir_name = "mass_plots6_tau2/"
#dir_name = "mass_plots6_2mM/"
dir_name = dir_name = "mass_plots_all_1.3mM/"


"""
#names for reference, defined above
unique_pres = np.unique(np.asarray(pre_types))
unique_posts = np.unique(np.asarray(post_types))

pre_mses = [[] for i in range(0, 12)] #create sub entries for each stim at each cre_type
post_mses = [[] for i in range(0, 12)] #create sub entries for each stim at each cre_type
"""


srp_means = {}
for i in range(1, len(sys.argv)):
    file_name = sys.argv[i]
    #print("printing file name")
    #rint(file_name)
    """
    pair_id = file_name.split(post_cut, 1)[1] #split on "full_" for sst
    #pair_id = file_name.split(post_cut, 1)[1]
    """
    pair_id = file_name.split('_')[5]
    pre_type= file_name.split('_')[3]
    pre_type = pre_type.split("/")[1]
    post_type= file_name.split('_')[4]
    #print("split1: "+pair_id)
    pair_id = pair_id.split(".p",1)[0]
    #print("split2: "+pair_id)
    pair_id = int(pair_id)
    pair_id_list.append((pair_id, pre_type, post_type))
    pkl1_file = open(file_name, "rb")
    params1 = pickle.load(pkl1_file)
    #set stand in vals for sigma params
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

for pair_id, pre_type, post_type in pair_id_list:
    keys = target_dict[pair_id].keys()
    protocols = []
    for key in keys:
        #print(key)
        if key != 'pair_ID':
            protocols.append(key)
        
    num_rows = int(math.ceil(len(protocols)/3))
    #print(protocols)
    #fig, axs = plt.subplots(num_rows, 3, figsize =(15, 4*num_rows) )
    #print(axs)
    #fig.suptitle(pre_type + " to "+ post_type+": "+str(pair_id), fontsize=16)
    if len(protocols)>1:
        for index in range(0, len(protocols)):
            #print("pair_ID: "+str(pair_id))
            x = int(index/3)
            y = index % 3
            #print("len(protocols) = "+str(len(protocols)))
            #print("x="+str(x)+" y="+str(y)+" key ="+str(protocols[index]))
            num_runs = len(target_dict[pair_id][protocols[index]])
            if target_dict[pair_id][protocols[index]].ndim < 2:
                #print("Dimension too low")
                continue
            title = "Protocol: " + str(protocols[index][0])+"Hz, "+ str(protocols[index][1])+"s, "+str(num_runs)+" runs"
            #print("printing protocol")
            #print(title)
            #conditional to select only one protocol for generation of line fits, comment out otherwise
            """
            if not ((protocols[index][0] == 50.0) and (protocols[index][1] >0.23) and (protocols[index][1] < 0.3)):
                print("removed "+title)
                continue
            """
            
            """
            if not (protocols[index][0] == 50.0):
                print("removed "+title)
                continue
            """
            
            #get MSE of means to 
            #does this operation make sense dimensionally?
            #squared_error =  math.pow(target_dict[pair_id][protocols[index]] - srp_means[pair_id][protocols[index]], 2) 
            
            if len(target_dict[pair_id][protocols[index]]) > 1: #multiple runs case
                errors = [[] for i in range(0, 12)]
                model_vals = [[] for i in range(0, 12)]
                raw_vals = [[] for i in range(0, 12)]
                raw_vals_all = [[] for i in range(0, 12)]
                #print("printing srp_means[pair_id][protocols[index]][2]")
                #print(srp_means[pair_id][protocols[index]][2])
                #print("target_dict[pair_id][protocols[index]][:,2]")
                #print(target_dict[pair_id][protocols[index]][:,2])
                for i in range(0, 12):
                    #errors[i] = np.square(target_dict[pair_id][protocols[index]][:,i] - srp_means[pair_id][protocols[index]][i])
                    errors[i] = np.square(np.nanmean(target_dict[pair_id][protocols[index]][:,i]) - np.nanmean(srp_means[pair_id][protocols[index]][i]))
                    model_vals[i] = np.nanmean(srp_means[pair_id][protocols[index]][i])
                    raw_vals[i] = np.nanmean(target_dict[pair_id][protocols[index]][:,i])
                    raw_vals_all[i] = target_dict[pair_id][protocols[index]][:,i]
                #print("printing errrors")
                #print(errors)
                errors = np.transpose(np.stack(errors))
                #print("printing pre_mses[pre_type]")
                #print(pre_mses[pre_type])
                #print("len(pre_mses[pre_type])")
                #print(len(pre_mses[pre_type]))
                #pre_fits
                if len(pre_mses[pre_type]) > 0:
                    pre_mses[pre_type] = np.append(pre_mses[pre_type], [errors], axis=0)
                    #print("printing appended")
                    #print(pre_mses[pre_type])
                    #print("length of appended:")
                    #print(len(pre_mses[pre_type]))
                else:
                    pre_mses[pre_type] = np.asarray([errors])
                    #pre_mses[pre_type] = errors
                    
                if len(pre_fits[pre_type]) > 0:
                    #print(pre_fits[pre_type])
                    pre_fits[pre_type]["model"] = np.append(pre_fits[pre_type]["model"], [model_vals], axis=0)
                    pre_fits[pre_type]["raw"] = np.append(pre_fits[pre_type]["raw"], [raw_vals], axis=0)
                    #pre_fits[pre_type]["raw_all"] = np.append(pre_fits[pre_type]["raw_all"], raw_vals_all, axis=0)
                    pre_fits[pre_type]["raw_all"] = np.concatenate((pre_fits[pre_type]["raw_all"], raw_vals_all), axis=1)
                else:
                    pre_fits[pre_type]["model"] = np.asarray([model_vals])
                    pre_fits[pre_type]["raw"] = np.asarray([raw_vals])
                    pre_fits[pre_type]["raw_all"] = np.asarray(raw_vals_all)
                
                if len(post_mses[post_type]) > 0:
                    post_mses[post_type] = np.append(post_mses[post_type], [errors], axis=0)
                else:
                    post_mses[post_type] = np.asarray([errors])
                    
                if len(post_fits[post_type]) > 0:
                    post_fits[post_type]["model"] = np.append(post_fits[post_type]["model"], [model_vals], axis=0)
                    post_fits[post_type]["raw"] = np.append(post_fits[post_type]["raw"], [raw_vals], axis=0)
                    #post_fits[post_type]["raw_all"] = np.append(post_fits[post_type]["raw_all"], [raw_vals_all], axis=0)
                    post_fits[post_type]["raw_all"] = np.concatenate((post_fits[post_type]["raw_all"], raw_vals_all), axis=1)
                else:
                    post_fits[post_type]["model"] = np.asarray([model_vals])
                    post_fits[post_type]["raw"] = np.asarray([raw_vals])
                    post_fits[post_type]["raw_all"] = np.asarray(raw_vals_all)
                    #post_mses[post_type] = errors
#slice arrays by column to get distributions within each cre_type
post_2_vals = []
pre_2_vals = []
post_2_labels = []
pre_2_labels = []
        
post_8_vals = []
pre_8_vals = []
post_8_labels = []
pre_8_labels = []


post_9_vals = []
pre_9_vals = []
post_9_labels = []
pre_9_labels = []

post_all_vals = []
pre_all_vals = []
post_all_labels = []
pre_all_labels = []

post_first8_vals = []
pre_first8_vals = []
post_first8_labels = []
pre_first8_labels = []

post_last4_vals = []
pre_last4_vals = []
post_last4_labels = []
pre_last4_labels = []

for post_type in post_mses.keys():
    if len(post_mses[post_type]) > 1:
        #print(post_mses[post_type])
        
        data_2 = post_mses[post_type][:,1]
        post_2_vals.append(data_2[~np.isnan(data_2)]) #get all vals in col (stim) 2 that aren't nan
        post_2_labels.append(cre_capital[post_type])
        
        data_8 = post_mses[post_type][:,7]
        post_8_vals.append(data_8[~np.isnan(data_8)]) #get all vals in col (stim) 2
        post_8_labels.append(cre_capital[post_type])
        
        data_9 = post_mses[post_type][:,8]
        post_9_vals.append(data_9[~np.isnan(data_9)]) #get all vals in col (stim) 2
        post_9_labels.append(cre_capital[post_type])
        
        #compute overall post
        data_all = np.ndarray.flatten(post_mses[post_type][:,:])
        post_all_vals.append(data_all[~np.isnan(data_all)]) #get all vals in col (stim) 2
        post_all_labels.append(cre_capital[post_type])
        
        #compute first 8 post
        data_first8 = np.ndarray.flatten(post_mses[post_type][:,:8])
        post_first8_vals.append(data_first8[~np.isnan(data_first8)]) #get all vals in col (stim) 2
        post_first8_labels.append(cre_capital[post_type])
        
        #compute overall post
        data_last4 = np.ndarray.flatten(post_mses[post_type][:,8:])
        post_last4_vals.append(data_last4[~np.isnan(data_last4)]) #get all vals in col (stim) 2
        post_last4_labels.append(cre_capital[post_type])

for pre_type in pre_mses.keys():
    if len(pre_mses[pre_type]) > 1:
        #print("pre_mses[pre_type]:")
        #print(pre_mses[pre_type])
        
        data_2 = pre_mses[pre_type][:,1]
        pre_2_vals.append(data_2[~np.isnan(data_2)]) #get all vals in col (stim) 2
        pre_2_labels.append(pre_type)
        
        data_8 = pre_mses[pre_type][:,7]
        pre_8_vals.append(data_8[~np.isnan(data_8)]) #get all vals in col (stim) 2
        pre_8_labels.append(pre_type)
        
        data_9 = pre_mses[pre_type][:,8]
        pre_9_vals.append(data_9[~np.isnan(data_9)]) #get all vals in col (stim) 2
        pre_9_labels.append(pre_type)
        
        #compute overall pre
        data_all = np.ndarray.flatten(pre_mses[pre_type][:,:])
        pre_all_vals.append(data_all[~np.isnan(data_all)]) #get all vals in col (stim) 2
        pre_all_labels.append(pre_type)
        
        #compute first 8
        data_first8 = np.ndarray.flatten(pre_mses[pre_type][:,:8])
        pre_first8_vals.append(data_first8[~np.isnan(data_first8)]) #get all vals in col (stim) 2
        pre_first8_labels.append(pre_type)
        
        #compute last 4
        data_last4 = np.ndarray.flatten(pre_mses[pre_type][:,8:])
        pre_last4_vals.append(data_last4[~np.isnan(data_last4)]) #get all vals in col (stim) 2
        pre_last4_labels.append(pre_type)
        
"""
#boxplot distributions by cre-type
plt.figure()
x_coordinates = [i for i in range(1, len(post_2_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_2_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit to Second Stimulation by Post-synaptic Cre_type")
plt.xticks(x_coordinates, post_2_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
plt.show()
"""      
  
save_dir = 'fig1/'

#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_8_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit to Eighth Simulation")
plt.xticks(x_coordinates, post_8_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
savetitle = save_dir+"MSE_to_Eighth_Simulation.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle)
#print("printing save location")
#print(savetitle)
#plt.show()

#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_9_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_9_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit to Ninth Simulation")
plt.xticks(x_coordinates, post_9_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
savetitle = save_dir+"MSE_to_Ninth_Simulation.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle)
#plt.show()


#boxplot distributions by cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_all_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_all_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Fit by Post-Type")
plt.xticks(x_coordinates, post_all_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
savetitle = save_dir+"MSE_of_Post_Fit.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(600)
f.tight_layout()
plt.savefig(savetitle)
#print("printing save location")
#print(savetitle)
#plt.show()

#boxplot distributions by first 8 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_first8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_first8_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Fit MSE of First 8 by Post-Type")
plt.xticks(x_coordinates, post_first8_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
savetitle = save_dir+"MSE_of_Post_Fit_first8.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)

#print fit means
all_first8_vals = []
all_first8_vals_not_sst =[]
sst_first8_vals = []
for i in range(0, len(post_first8_vals)):
    print("first 8 vals")
    print("mean MSE for "+post_first8_labels[i]+" post = "+str(np.mean(np.asarray(post_first8_vals[i]))))
    print("SEM of mean MSE for "+post_first8_labels[i]+" post = "+str(stat.sem(np.asarray(post_first8_vals[i]))))
    all_first8_vals = all_first8_vals+[val for val in post_first8_vals[i]]
    if post_first8_labels[i] == 'Sst':
    #if post_first8_labels[i] == 'Tlx3':
        print("adding sst")
        sst_first8_vals  = post_first8_vals[i]
    else:
        all_first8_vals_not_sst = all_first8_vals_not_sst+[val for val in post_first8_vals[i]]
    

print("testing sst first 8 t-test vs all")
#print(np.asarray(sst_first8_vals))
#print(np.asarray(all_first8_vals_not_sst))
print(ttest_ind(np.asarray(sst_first8_vals), np.asarray(all_first8_vals_not_sst)))
    
print("mean MSE across all first 8: "+str(np.mean(np.asarray(all_first8_vals))))
print("SEM of mean MSE across all first 8: "+str(stat.sem(np.asarray(all_first8_vals))))

#boxplot distributions by last 4 cre-type
f= plt.figure()
x_coordinates = [i for i in range(1, len(post_last4_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(post_last4_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("Fit MSE of Last 4 by Post-Type")
plt.xticks(x_coordinates, post_last4_labels)
plt.xlabel("Representation")
plt.ylabel("MSE")
#plt.ylim([0,1])
savetitle = save_dir+"MSE_of_Post_Fit_last4.svg"
f.set_size_inches((4.57, 1.97))
f.set_dpi(1200)
f.tight_layout()
plt.savefig(savetitle, transparent=True)

#print fit means
all_last4_vals = []
all_last4_vals_not_sst =[]
sst_last4_vals = []
for i in range(0, len(post_last4_vals)):
    print("Last 4 vals")
    print("mean MSE for "+post_last4_labels[i]+" post = "+str(np.mean(np.asarray(post_last4_vals[i]))))
    print("SEM of mean MSE for "+post_last4_labels[i]+" post = "+str(stat.sem(np.asarray(post_last4_vals[i]))))
    all_last4_vals = all_last4_vals+[val for val in post_last4_vals[i]]
    if post_last4_labels[i] == 'Sst':
    #if post_first8_labels[i] == 'Tlx3':
        print("adding sst")
        sst_last4_vals  = post_last4_vals[i]
    else:
        all_last4_vals_not_sst = all_last4_vals_not_sst+[val for val in post_last4_vals[i]]

print("testing sst last 4 t-test vs all")
print(ttest_ind(np.asarray(sst_last4_vals), np.asarray(all_last4_vals_not_sst)))

print("mean MSE across all last 4: "+str(np.mean(np.asarray(all_last4_vals))))
print("SEM of mean MSE across all last 4: "+str(stat.sem(np.asarray(all_last4_vals))))
"""
#boxplot distributions by cre-type
plt.figure()
x_coordinates = [i for i in range(1, len(pre_2_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(pre_2_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit to Second Stimulation by Pre-synaptic Cre_type")
plt.xticks(x_coordinates, pre_2_labels)
plt.xlabel("Cre_type")
plt.ylabel("MSE")
#plt.ylim([0,1])
plt.show()
        
#boxplot distributions by cre-type
plt.figure()
x_coordinates = [i for i in range(1, len(pre_8_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(pre_8_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit to Eighth Stimulation by Pre-synaptic Cre_type")
plt.xticks(x_coordinates, pre_8_labels)
plt.xlabel("Cre_type")
plt.ylabel("MSE")
#plt.ylim([0,1])
plt.show()

#boxplot distributions by cre-type
plt.figure()
x_coordinates = [i for i in range(1, len(pre_9_vals)+1)]
ax = plt.subplot(111)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.bar(x_coordinates, y_vals, yerr=y_errors)
plt.boxplot(pre_9_vals)
#plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
plt.title("MSE of Model Fit To Ninth (Recovery) Stimulation by Pre-synaptic Cre_type")
plt.xticks(x_coordinates, pre_9_labels)
plt.xlabel("Cre_type")
plt.ylabel("MSE")
#plt.ylim([0,1])
plt.show()
        
"""

#--------------------------------------------------
#plot model and mse by stim for each cre_type

for post_type in post_mses.keys():
    if len(post_mses[post_type]) > 1: #normally 4
        #print("post_type = "+post_type)
        #print(post_mses[post_type])
        model_vals = []
        raw_vals = []
        raw_sds = []
        raw_sems = []
        for i in range(0, 12):
            model_data = post_fits[post_type]["model"][:, i]
            mean = np.nanmean(model_data)
            model_vals.append(mean)
            
            raw_data = post_fits[post_type]["raw"][:, i]
            raw_data_all = post_fits[post_type]["raw_all"][:, i]
            raw_mean = np.nanmean(raw_data)
            raw_sd = np.nanstd(raw_data)
            raw_vals.append(raw_mean)
            raw_sds.append(raw_sd)
            #raw_sem = stat.sem(raw_data, axis=None, ddof=0, nan_policy='omit')
            raw_sem = stat.sem(raw_data_all, axis=None, ddof=0, nan_policy='omit')
            raw_sems.append(raw_sem)
            #print("printing sem")
            #print(raw_sem)
            #vals.append(post_mses[post_type][:, i])
        x_coordinates = [i for i in range(1, len(model_vals)+1)]
        x_coordinates_1 = [i for i in range(1, 9)]
        x_coordinates_2 = [i for i in range(9, 13)]
        #plt.figure()
        #ax = plt.subplot(111)
        f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
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
        
        #print("printing raw_sems")
        #print(raw_sems)
        #plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data") #changed to sem
        ax.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7) 
        ax2.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)#changed to sem
        
        ax.scatter(x_coordinates, raw_vals, color="black") 
        ax2.scatter(x_coordinates, raw_vals, color="black") 
        
        ax.plot(x_coordinates, model_vals, label="Model", color="#cc3311")
        ax2.plot(x_coordinates, model_vals, label="Model", color="#cc3311") 
        
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
        f.suptitle(cre_capital[post_type]+" Post")
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
        if post_type == "sst": 
            pickel_file = fig_folder+"sst_plot_means"
            with open(pickel_file, 'wb') as handle:
                pickle.dump((raw_vals, raw_sems), handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_name = fig_folder+"model_data_fit_post_"+post_type+".svg"
        f.tight_layout()
        plt.savefig(save_name)
        #plt.show()
        
        
for pre_type in pre_mses.keys():
    if len(pre_mses[pre_type ]) > 1: #normally 4
        #print("post_type = "+pre_type )
        #print(post_mses[pre_type ])
        model_vals = []
        raw_vals = []
        raw_sds = []
        raw_sems = []
        for i in range(0, 12):
            model_data = pre_fits[pre_type ]["model"][:, i]
            mean = np.nanmean(model_data)
            model_vals.append(mean)
            
            raw_data = pre_fits[pre_type ]["raw"][:, i]
            raw_data_all = pre_fits[pre_type ]["raw_all"][:, i]
            raw_mean = np.nanmean(raw_data)
            raw_sd = np.nanstd(raw_data)
            #raw_sem = stat.sem(raw_data, axis=None, ddof=0, nan_policy='omit')
            raw_sem = stat.sem(raw_data_all, axis=None, ddof=0, nan_policy='omit')
            raw_sems.append(raw_sem)
            raw_vals.append(raw_mean)
            raw_sds.append(raw_sd)
            #print("printing sem")
            #print(raw_sem)
            #vals.append(post_mses[post_type][:, i])dists
        """
        x_coordinates = [i for i in range(1, len(model_vals)+1)]
        plt.figure()
        ax = plt.subplot(111)
        #f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #ax2.spines['top'].set_visible(False)
        #ax2.spines['right'].set_visible(False)
        #plt.bar(x_coordinates, y_vals, yerr=y_errors)
        plt.plot(x_coordinates, model_vals, label="Model")
        #ax.plot(x_coordinates, model_vals, label="Model")
        #ax2.plot(x_coordinates, model_vals, label="Model")
        
        plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems , label="Data") #changed to use SEM
        #ax.errorbar(x_coordinates, raw_vals, yerr=raw_sems , label="Data") #changed to use SEM
        #ax2.errorbar(x_coordinates, raw_vals, yerr=raw_sems , label="Data") #changed to use SEM
        
        #ax.set_xlim(0, 8.1)  # outliers only
        #x2.set_xlim(8.9,13)  # most of the data
        
        #ax.spines['right'].set_visible(False)
        #ax2.spines['left'].set_visible(False)
        print(x_coordinates)
        print(raw_vals)
        #plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems , label="Data") #changed to use SEM
        #plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
        plt.title("Mean Value of Model and Data by Stimulus for "+pre_type +" Pre_type")
        plt.legend()
        #plt.xticks(x_coordinates, x_coordinates)
        plt.xlabel("Stimulus #")
        plt.ylabel("Normalized Amplitude")
        #plt.ylim([0,1])
        plt.show()
        """
        x_coordinates = [i for i in range(1, len(model_vals)+1)]
        x_coordinates_1 = [i for i in range(1, 9)]
        x_coordinates_2 = [i for i in range(9, 13)]
        #plt.figure()
        #ax = plt.subplot(111)
        f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', gridspec_kw={'width_ratios': [2, 1]})
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
        
        #print("printing raw_sems")
        #print(raw_sems)
        #plt.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data") #changed to sem
        ax.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7) 
        ax2.errorbar(x_coordinates, raw_vals, yerr=raw_sems, label="Data", color="black",
        ls="dashed", capsize=capsize, elinewidth=0.7)#changed to sem
        
        ax.scatter(x_coordinates, raw_vals, color="black") 
        ax2.scatter(x_coordinates, raw_vals, color="black") 
        
        ax.plot(x_coordinates, model_vals, label="Model", color="#cc3311")
        ax2.plot(x_coordinates, model_vals, label="Model", color="#cc3311") 
        
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
        #f.suptitle("Model Fit For "+pre_type +" Pre_type")
        f.suptitle("Mean of "+cre_capital[post_type]+" Pre")
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
        f.text(0.5, 0.02, 'Stimulus #', ha='center')
        ax.set_ylabel("Normalized Amplitude") 
        
        #plt.ylim([0,1])
        save_name = fig_folder+"model_data_fit_pre_"+pre_type+".svg"
        f.tight_layout()
        plt.savefig(save_name)
        #plt.show()
#--------------------------------------------------
#plot mse by stim for each cre_type

"""
for post_type in post_mses.keys():
    if len(post_mses[post_type]) > 1: #normally 4
        #print("post_type = "+post_type)
        #print(post_mses[post_type])
        vals = []
        for i in range(0, 12):
            data =post_mses[post_type][:, i]
            filtered_data = data[~np.isnan(data)]
            vals.append(filtered_data)
            #vals.append(post_mses[post_type][:, i])
        x_coordinates = [i for i in range(1, len(vals)+1)]
        plt.figure()
        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.bar(x_coordinates, y_vals, yerr=y_errors)
        plt.boxplot(vals)
        #plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
        plt.title("MSE of Model Fit by Stimulus For "+post_type+" Post_type")
        plt.xticks(x_coordinates, x_coordinates)
        plt.xlabel("Stimulus #")
        plt.ylabel("MSE")
        #plt.ylim([0,1])
        plt.show()
        
for pre_type in pre_mses.keys():
    if len(pre_mses[pre_type]) > 1:  #normally 4
        vals = []
        for i in range(0, 12):
            data =pre_mses[pre_type][:, i]
            filtered_data = data[~np.isnan(data)]
            vals.append(filtered_data)
            #vals.append(pre_mses[pre_type][:, i])
        x_coordinates = [i for i in range(1, len(vals)+1)]
        plt.figure()
        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.bar(x_coordinates, y_vals, yerr=y_errors)
        plt.boxplot(vals)
        #plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
        plt.title("MSE of Model Fit by Stimulus For "+pre_type+" Pre_type")
        plt.xticks(x_coordinates, x_coordinates)
        plt.xlabel("Stimulus #")
        plt.ylabel("MSE")
        #plt.ylim([0,1])
        plt.show()
"""

        
        #generate the plotting dims
        #boxplot code
"""
        plt.figure()
        ax = plt.subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #plt.bar(x_coordinates, y_vals, yerr=y_errors)
        plt.boxplot(y_vals)
        #plt.title("Supervised Multilabel Classification Accuracy By Representation With PCA \n Non-Baseline Kruskal Wallis "+pval_to_str(group_pval))
        plt.title("Supervised Multilabel Classification Accuracy By Representation")
        plt.xticks(x_coordinates, x_labels)
        plt.xlabel("Representation")
        plt.ylabel("Mean Accuracy")
        #plt.ylim([0,1])
        plt.show()
        """
        
        #then you have succeded!
        
        #don't plot if only one run
"""
        if len(protocols) > 2:
            plot_mufit3(axs[x, y], target_dict[pair_id], srp_means[pair_id], protocols[index])
            axs[x, y].set_title(title)
        else:
            plot_mufit3(axs[y], target_dict[pair_id], srp_means[pair_id], protocols[index])
            axs[y].set_title(title)
        """
    #fig.tight_layout() 
    
    #saving code
"""
    save_name = dir_name + pre_type + "_"+ post_type + "_" + str(pair_id)
    plt.savefig(save_name)
    #plt.show()
    """
    
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

