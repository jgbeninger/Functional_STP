import numpy as np
import scipy.stats
import aisynphys
from aisynphys.database import SynphysDatabase
from aisynphys.cell_class import CellClass, classify_cells, classify_pairs
from aisynphys.dynamics import *
from sqlalchemy.orm import aliased

import pandas
import pickle

# SET CACHE FILE LOCATION FOR DATASET DOWNLOAD:
aisynphys.config.cache_path = "/tungstenfs/scratch/gzenke/rossjuli/datasets"

# WARNING: DOWNLOADS THE FULL 180 GB DATASET
db = SynphysDatabase.load_version('synphys_r1.0_2019-08-29_full.sqlite')

# Load all synapses associated with mouse V1 projects
pairs = db.pair_query(
    synapse=True,
    species='mouse',
    synapse_type='ex',
    #acsf='2mM Ca & Mg',  # '1.3mM Ca & 1mM Mg' or '2mM Ca & Mg'
    acsf='1.3mM Ca & 1mM Mg', 
    electrical=False  # Exclude gap junctions
).all()

results = pandas.DataFrame(columns=['pair_id',
                                    'pre_cell',
                                    'post_cell',
                                    'rec_id',
                                    'clamp_mode',
                                    'ind_freq',
                                    'rec_delay',
                                    'amps',
                                    'psc_amp',
                                    'psp_amp']
                           )
#create set of cre types for later print out
unique_cre_types = set()

for ix in range(len(pairs)):
    print('processing pair {} of {}'.format(ix + 1, len(pairs)))
    pair = pairs[ix]
    pr_recs = pulse_response_query(pair).all()  # extract all response trains

    # cull out all PRs that didn't get a fit
    pr_recs = [pr_rec for pr_rec in pr_recs if pr_rec.PulseResponseFit.fit_amp is not None]
    # sort by clamp mode and frequency
    sorted = sorted_pulse_responses(pr_recs)

    if sorted != {}:  # if any recordings have a fit

        # Loop through all recordings
        for key, recs in sorted.items():
            clamp_mode, ind_freq, rec_delay = key

            # only consider voltage clamp recordings
            # if clamp_mode != 'vc':
            #    continue

            for recording, pulses in recs.items():
                if 1 not in pulses or 2 not in pulses:
                    continue
                amps = {k: r.PulseResponseFit.fit_amp for k, r in pulses.items()}
                if clamp_mode == 'vc':
                    if not isinstance(pair.synapse.psc_amplitude, float):
                        continue
                    #normamps = np.array(list(amps.values())) / pair.synapse.psc_amplitude
                    first_val = list(amps.values())[0]
                    print(first_val)
                    #normamps = np.array(list(amps.values())) / first_val
                    normamps = np.array(list(amps.values())) 

                elif clamp_mode == 'ic':
                    if not isinstance(pair.synapse.psp_amplitude, float):
                        continue
                    #normamps = np.array(list(amps.values())) / pair.synapse.psp_amplitude
                    first_val = list(amps.values())[0]
                    print(first_val)
                    #normamps = np.array(list(amps.values())) / first_val
                    normamps = np.array(list(amps.values()))
                
                #create set of all cre_types for later printout
                unique_cre_types.add(pair.pre_cell.cre_type)
                unique_cre_types.add(pair.post_cell.cre_type)
                
                results = results.append({'pair_id': pair.id,
                                          'pre_cell': pair.pre_cell.cre_type, #previosuly .cell_class
                                          'post_cell': pair.post_cell.cre_type,
                                          'rec_id': recording.id,
                                          'clamp_mode': clamp_mode,
                                          'ind_freq': ind_freq,
                                          'rec_delay': rec_delay,
                                          'amps': np.array(list(amps.values())),
                                          'normamps': normamps,
                                          'stimuli': list(amps.keys()),
                                          'psc_amp': pair.synapse.psc_amplitude,
                                          'psp_amp': pair.synapse.psp_amplitude},
                                         ignore_index=True)


#JB: take the "results" dataframe and process it into a dict organized by synapse type
def organize_syn_dict(pulse_res_df):
    print(str(pulse_res_df))
    syn_dict = {}
    for index, row in results.iterrows():
        #bin rec_delay values, note this edits the input dataframe so it should only be run once per
        #use of the script, fix this for future
        if row['rec_delay'] in [0.125, 0.126, 0.127, 0.128]:
            row['rec_delay'] = 0.125
        if row['rec_delay'] in [0.250,0.251, 0.252, 0.253]:
            row['rec_delay'] = 0.250
        syn_key = (row['pre_cell'], row['post_cell'])
        sim_key = (row['clamp_mode'], row['ind_freq'], row['rec_delay'])

        processed_amps = fill_pad(row['normamps'], row['stimuli'])
        if (syn_key in syn_dict):
            if row['pair_id'] in syn_dict[syn_key]: #need pair_IDs here
                if sim_key in syn_dict[syn_key][row['pair_id']]:
                    print("on row number: "+str(index))
                    print(processed_amps)
                    print(syn_dict[syn_key][row['pair_id']][sim_key])
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.vstack((syn_dict[syn_key][row['pair_id']][sim_key] , processed_amps))
                else:
                    syn_dict[syn_key][row['pair_id']][sim_key] = np.array([processed_amps])
            else:
                syn_dict[syn_key][row['pair_id']]= {sim_key: np.array([processed_amps])}
        else:
            syn_dict[syn_key] = {row['pair_id']: {sim_key: np.array([processed_amps])}} 
            syn_dict[syn_key]['pair_IDs'] = set()
        syn_dict[syn_key]['pair_IDs'].add(str(row['pair_id']))
    return syn_dict

#create array of 12 nan values and add in values from norm_amp_arr at 
#locations of corresponding keys
def fill_pad(norm_amp_arr, arr_keys):
    arr = np.empty((12))*np.nan
    num_allocated = 0
    for key in arr_keys:
        print("key is:")
        print(key)
        arr[key-1] = norm_amp_arr[num_allocated]
        num_allocated = num_allocated + 1
    return arr
    


def process_amps(amp_arr, clamp_mode, vc_amp, ic_amp, normalize=True):
    if normalize:
        if (clamp_mode == 'ic'): #normalize current
            if ic_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
            print("ic_amp ="+str(ic_amp))
            out_arr = np.divide(out_arr, ic_amp)
        elif(clamp_mode == 'vc'): #normalize voltage
            if vc_amp == None:
                return (False, None)
            out_arr = pad_amps(amp_arr)
            out_arr = np.divide(out_arr, vc_amp)
        else:
            raise NameError('Unknown clamp_mode')
    else:
        out_arr = pad_amps(amp_arr)
    return (True, out_arr)

def pad_amps(amp_list):
    extracted_amplitudes = np.zeros(12)
    for i in range(12):
        if i in range(len(amp_list)):
            extracted_amplitudes[i] = amp_list[i]
        else:
            extracted_amplitudes[i] = np.nan
    return extracted_amplitudes


def calc_norm_amplitudes(results):

    initialAmp = {}
    # iterate over pairs
    for id in list(results.pair_id.unique()):
        subdat = results[results.pair_id==id]
        vcdat = subdat[subdat.clamp_mode == 'vc']
        icdat = subdat[subdat.clamp_mode == 'ic']
        initialAmp[id] = {'vc': np.array([vcdat.amps.iloc[i][0] for i in range(len(vcdat))]).mean(),
                          'ic': np.array([icdat.amps.iloc[i][0] for i in range(len(icdat))]).mean()}

    normAmp = []

    for i in range(len(results)):
        s = results.iloc[i]
        normAmp.append(np.array(s.amps) / initialAmp[s.pair_id][s.clamp_mode])

    results['normAmps'] = normAmp

    return results

def get_amps_first8(results, normalized=False):
    """
    Extracts response to first 8 pulses (no recovery pulses)
    """

    x = np.zeros((len(results), 8))

    for ix in range(len(results)):
        if normalized:
            amps = results.iloc[ix].normAmps
        else:
            amps = results.iloc[ix].amps

            if len(amps) >= 8:
                x[ix, :] = amps[:8]
            else:
                x[ix, :] = np.pad(amps, (0, 8-len(amps)), constant_values=np.nan)

    return x

print(unique_cre_types)


test_dict = organize_syn_dict(results)
pickle.dump(test_dict, open("Extracted_STP_1.3mM_Rodent.p", "wb"))
print(test_dict)
