#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:30:57 2023

@author: john
"""

import pickle
import sys
import numpy as np

#script to do input processing and save to permanent pickle 

#----------------------------------------------------------------------------------

#load in code for dictionaries
row_info = []
data = []
pyr_types = ['nr5a1', 'rorb']
pyr_indices = []

l5et_types = ['sim1', 'fam84b']
l5et_indices = []

data_post_pre = {}
unique_pre = set()
unique_post = set()
unique_pairs = set()

measures_file = open('Measures_ex_1.3mM_Human_Type_Pair_Stim.p', "rb")
#measures_file = open('Measures_ex_2mM_Mouse_Type_Pair_Stim.p', "rb")
#print('printing measures dict:')
measures_dict = pickle.load(measures_file)
#print(measures_dict)
num_100s = 0
for i in range(1, len(sys.argv)):
    cut_on = '_'
    file_name = sys.argv[i]
    #print(i)
    #print(file_name)
    pair_id = file_name.split(cut_on)[5] #6 for standard #8 for two phase, 9 for windows
    pair_id = int(pair_id.split(".p")[0])
    pre_type = file_name.split(cut_on)[3] #4 for standard 6 for two phase, 7 for windows
    pre_type = pre_type.split(cut_on)[0]
    pre_type = pre_type.split("/")[1]
    post_type = file_name.split(cut_on)[4] #5 for standard 7 for two phase, 8 for windows
    post_type = post_type.split(cut_on)[0]
    #print("Pre_type = "+pre_type+" post_type = "+post_type+" pair_id = "+str(pair_id))
    pickle_file = open(file_name, "rb")
    params = pickle.load(pickle_file)
    mu_baseline, mu_amps, mu_taus, SD, mu_scale = params
    new_row = []
    #add identifiers
    new_row.append(pre_type) #1
    new_row.append(post_type) #2
    new_row.append(pair_id) #3
    
    unique_pre.add(pre_type)
    unique_post.add(post_type)
    unique_pairs.add(pre_type+post_type)
    
    #add measures: phys 2 version
    """
    try:
        print('attempted key = ')
        print((pre_type, post_type))
        print("keys = ")
        print(measures_dict[(pre_type, post_type)][pair_id].keys())
        
        key_50hz = ('ic', 50.0, 0.25)

        #select only trials with 100hz data for comparison
        key_100hz = ('ic', 100.0, 0.25)
        key_50hz = ('ic', 50.0, 0.25)
        if key_100hz in measures_dict[(pre_type, post_type)][pair_id].keys():
            num_100s += 1
            print("100hz entry: "+str(num_100s))
        else:
            print("no 100")
            continue
        
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        print("printing ppr:" +str(ppr))
        #skip the excessively high ppr
        
        if ppr > 4: #1000 for the excessive value
            print("skipper ppr = "+str(ppr))
            continue
       
        areas = measures_dict[(pre_type, post_type)][pair_id]['areas_50hz_mean']
        print("printing area:" +str(areas))
        #this first fifth is having trouble, values are clearly wrong (way too high)
        #replacement of min and max values needs to be fixed in extraction code
        release_prob = measures_dict[(pre_type, post_type)][pair_id]['release_prob_all']
        print("printing releaase_prob:" +str(release_prob))
        #print("printing first_fifth:" +str(first_fifth))
        first_fifth = measures_dict[(pre_type, post_type)][pair_id]['first_fifth_50hz_mean']
        first_second = measures_dict[(pre_type, post_type)][pair_id]['first_second_50hz_mean']
        #new_row.append(ppr) #4
        new_row.append(areas) #5
        new_row.append(release_prob) #6
        #new_row.append(first_fifth) #7
        new_row.append(first_second) #8
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id][key_50hz ]['recovery'] 
        new_row.append(recovery_50) #9
        

        #100hz measures

        areas_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['area_first_eight_mean'] #10
        ppr_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['first_second_mean'] #11
        first_fifth_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['first_fifth_mean'] #12
        recovery_100 = measures_dict[(pre_type, post_type)][pair_id][key_100hz ]['recovery'] #13
        new_row.append(areas_100) #10
        new_row.append(ppr_100) #11
        #new_row.append(first_fifth_100) #12
        new_row.append(recovery_100) #13

        
    except:
        print("triggered exception")
        #adjust this so it does not add all missing rows if only some are missing
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
    """
    
    #add measures
    try:
        key_50hz = ('ic', 50.0, 0.25)
        #print('attempted key = ')
        #print((pre_type, post_type))
        #print("keys = ")
        #print(measures_dict[(pre_type, post_type)][pair_id].keys())
        ppr = measures_dict[(pre_type, post_type)][pair_id]['Paired_Pulse_50Hz']
        #print("printing ppr:" +str(ppr))
        #skip the excessively high ppr
        
       
        if ppr > 4: #1000 for the excessive value, note: this was set to 4 until March 9th when changed for some testing, 4 may still be the best value
            print("skipper ppr = "+str(ppr))
            continue
      
        
        areas = measures_dict[(pre_type, post_type)][pair_id]['areas_50hz_mean']
        #print("printing area:" +str(areas))
        #this first fifth is having trouble, values are clearly wrong (way too high)
        #replacement of min and max values needs to be fixed in extraction code
        release_prob = measures_dict[(pre_type, post_type)][pair_id]['release_prob_all']
        #print("printing releaase_prob:" +str(release_prob))
        #print("printing first_fifth:" +str(first_fifth))
        first_fifth = measures_dict[(pre_type, post_type)][pair_id]['first_fifth_50hz_mean']
        first_second = measures_dict[(pre_type, post_type)][pair_id]['first_second_50hz_mean']
        recovery_50 = measures_dict[(pre_type, post_type)][pair_id][key_50hz ]['recovery']
        #new_row.append(ppr)
        new_row.append(areas)
        new_row.append(release_prob)
        new_row.append(first_fifth)
        new_row.append(first_second)
        new_row.append(recovery_50)

    except:
        #adjust this so it does not add all missing rows if only some are missing
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
        new_row.append('missing_entry')
    
    
    #add model params
    new_row.append(mu_baseline)
    new_row = new_row + [mu_amp for mu_amp in mu_amps]
    new_row.append(SD)
    #new_row = new_row + [mu_tau for mu_tau in mu_taus]
    #new_row.append(sigma_baseline)
    #new_row = new_row +[sigma_amp for sigma_amp in sigma_amps]
    #new_row = new_row + [sigma_tau for sigma_tau in sigma_taus]
    #note: mu_scale=None and therefore not included
    #new_row.append(mu_scale) #just added mu scale for some reason?
    #print("printing mu_scale")
    #print(mu_scale)
    #new_row.append(sigma_scale)
    data.append(new_row)
    #print(params)[row[1] for row in cell_label_sorted[i]]
    #print("Printing new_row")
    #print(new_row)
    if post_type in data_post_pre:
        if pre_type in data_post_pre[post_type]:
            curr_data = data_post_pre[post_type][pre_type]
            data_post_pre[post_type][pre_type] = np.vstack((curr_data, np.array(new_row)))
        else:
             data_post_pre[post_type][pre_type] = np.array(new_row)
    else:
        data_post_pre[post_type] = {pre_type: np.array(new_row)}

#print(data)
num_cols = len(data[0])
#print(data[:,3:])
#params_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
#params_data = [row[3:] for row in data[:]] #model and phys
params_data = [row[8:] for row in data[:]] #model only

phys_data = [row[3:8] for row in data[:]] #phys only w/ 50Hz rec
hybrid_data = [row[3:] for row in data[:]] #model and phys
model_data = [row[8:] for row in data[:]] #model only
#model_data = [row[8:12] for row in data[:]] #only mu kernel data
#model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "sigma_baseline", "sigma_amp1", "sigma_amp2", "sigma_amp3", "sigma_scale"]
model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3", "SD", "mu_scale"]
#model_labels = ["mu_baseline", "mu_amp1", "mu_amp2", "mu_amp3"]

phys_labels = ["areas", "release_prob", "STP induction", "PPR", "50Hz Recovery"]

print("printing model data[0]")
print(model_data[0])
#params_data = [row[3:8] for row in data[:]]
#params_data = [row[3:10] for row in data[:]]


#params_data = [row[8:] for row in data[:]]
#row_labels = [row[0:8] for row in data[:]]
row_labels = [row[0:3] for row in data[:]]
params_arr = np.array(params_data)

#--------------------------------------------------------------------------

#human output pickle name
filename = "processed_input_human.p"
with open(filename, 'wb') as pickle_file:
    pickle.dump(data, pickle_file)
