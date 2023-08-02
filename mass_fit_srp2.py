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
from scipy.optimize import shgo

# Models
from srplasticity.tm import fit_tm_model, TsodyksMarkramModel
from srplasticity.srp import (
    ExpSRP,
    DetSRP,
    ExponentialKernel,
    _convolve_spiketrain_with_kernel,
    get_stimvec,
)
from srplasticity.inference import *
#--------------------------------------------------------------------
#loading data 

#srp fit parameters
#SRP model time constants
#mu_kernel_taus = [15, 100, 300]  #orginally [15, 200, 650] for both mu an sigma other 15, 100, 600
#sigma_kernel_taus = [15, 100, 300] #orginally [15, 200, 650] for both mu an sigma
color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}

# Initial guess for sigma scale
sigma_scale = 4

# Parameter ranges for grid search. Total of 128 initial starts
srp_param_ranges = (
    slice(-3, 1, 0.25),  # both baselines default: (-3,1,0.25)
    slice(-2, 2, 0.25),  # all amplitudes (weighted by tau in fitting procedure) defaul: (-2,2,0.25)
)
#--------------------------------------------------------------------

"""50] for both mu an sigma other 15, 100, 600
sigma_kernel_taus = [15, 100, 300] #orginally [15, 200, 650] for both mu an sigma
color = {"tm": "#0077bb", "srp": "#cc3311", "accents": "grey"}

target_dict = {}
for key in stimulus_dict:
    target_dict[key] = load_pickle(
        Path(data_dir / str(key + "_normalized_by_cell.pkl"))
    )
    # set zero values to nan
    target_dict[key][target_dict[key] == 0] = np.nan
print(target_dict)
"""

# Noise correlation data
'''
noisecor_data = {}
for key in stimulus_dict:
    noisecor_data[key] = load_pickle(
        Path(data_dir / "noisecorrelations" / str(key + ".pkl"))
    )
'''
# example trace
#example_trace = load_pickle(Path(data_dir / "example_trace.pkl"))

#--------------------------------------------------------------------

#note: copied as is from srplasticity:inference.py may need adjustment
def _default_parameter_bounds(mu_taus, sigma_taus):
    """ returns default parameter boundaries for the SRP fitting procedure """
    return [
        (-4.6, 6),  # mu baseline
        #*[(-10* tau, 10* tau) for tau in mu_taus],  # mu amps, originall7 *-10, *10
        *[(-150, 150), (-1000, 1001), (-3000, 3000)], #hand specify mu kernel bounds #stable supervised
        #*[(-150, 150), (-750, 750), (-1000, 1001), (-3000, 3000)],
        (-6, 6),  # sigma baseline
        *[(-10 * tau, 10 * tau) for tau in sigma_taus],  # sigma amps
        (0.001, 100),  # sigma scale
    ]



def _nll(y, mu, sigma):
    """
    Negative Log Likelihood
    :param y: (np.array) set of amplitudes
    :param mu: (np.array) set of means
    :param sigma: (np.array) set of stds
    """

    return np.nansum(
        (
            (y * mu) / (sigma ** 2)
            - ((mu ** 2 / sigma ** 2) - 1) * np.log(y * (mu / (sigma ** 2)))
            + np.log(gamma(mu ** 2 / sigma ** 2))
            + np.log(sigma ** 2 / mu)
        )
    )

def mse_loss(target_vals, mean_predicted):
    """
    Stand in Mean Squared error for training loss in first phase
    :param target_vals: (np.array) set of amplitudes
    :param mean_predicted: (np.array) set of means
    """
    loss = []
    alt_loss = [] #for defining loss as distance from mean of runs
    for key in target_vals.keys():
        vals_by_amp = [[] for i in range(0, 12)] #2d list for vals by amp
        for i in range(0, len(target_vals[key])):
            run_arr = target_vals[key][i] #get amplitudes from a single run
            run_err = []
            #print("run_arr = "+str(run_arr))
            
            if not np.isscalar(run_arr):
                for j in range(0, len(run_arr)):
                    vals_by_amp[j].append(run_arr[j])
                    run_err.append(math.pow((run_arr[j]-mean_predicted[key][j]), 2))
                    #run_err.append(math.pow(math.pow((run_arr[j]-mean_predicted[key][j]), 2), 0.5))
                #loss.append(np.nanmean(run_err))
                loss.append(run_err)
            
        """
        #alternate loss
        alt_run_err = []
        for i in range(0, len(vals_by_amp)):
            for j in range(0, len(run_arr)):
                #take mean to compare against
                #true_amp_mean = np.mean(vals_by_amp[i])
                
                #try to compare against median
                true_amp_mean = np.median(vals_by_amp[i])
                
                #alt_run_err.append(math.pow(math.pow((true_amp_mean-mean_predicted[key][j]), 2), 0.5))
                alt_run_err.append(math.pow((true_amp_mean-mean_predicted[key][j]), 2))
        alt_loss.append(np.nanmean(alt_run_err))
        """
        
    #standard version
    #"""
    print("loss= "+str(np.nanmean(loss)))
    #loss = np.asarray(loss)
    #loss = loss[abs(loss - np.mean(loss)) < 3 * np.std(loss)]
    return np.nanmean(loss)
    
    #"""
    """
    print("alt loss= "+str(np.nanmean(alt_loss)))
    return np.nanmean(alt_loss)
    """

def _convert_fitting_params(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1 : 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps : 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_baseline,
        mu_amps,
        mu_taus,
        sigma_baseline,
        sigma_amps,
        sigma_taus,
        mu_scale,
        sigma_scale,
    )

#--------------------------------------------------------------------

def _convert_fitting_params_det(x, mu_taus, sigma_taus, mu_scale=None):
    """
    Converts a vector of parameters for fitting `x` and independent variables
    (time constants and mu scale) to a vector that can be passed an an input
    argument to `ExpSRP` class
    """

    # Check length of time constants
    nr_mu_exps = len(mu_taus)
    nr_sigma_exps = len(sigma_taus)

    # Unroll list of initial parameters
    mu_baseline = x[0]
    mu_amps = x[1 : 1 + nr_mu_exps]
    sigma_baseline = x[1 + nr_mu_exps]
    sigma_amps = x[2 + nr_mu_exps : 2 + nr_mu_exps + nr_sigma_exps]
    sigma_scale = x[-1]

    return (
        mu_amps, #note: this order is the reverse of input to probabilistic model, correct the inconsistency in package
        mu_baseline
    )
#--------------------------------------------------------------------

def _objective_function(x, *args, phase=0):
    """
    Objective function for scipy.optimize.minimize
    :param x: parameters for SRP model as a list or array:
                [mu_baseline, *mu_amps,
                sigma_baseline, *sigma_amps, sigma_scale]
    :param phase: 0 indicates fitting only mu amps and mu baseline for fixed 
                    sigmas, 1 indicates fitting sigma params for fixed mu params
    :param args: target dictionary and stimulus dictionary
    :return: total loss to be minimized
    """
    # Unroll arguments
    #target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, phase = args
    target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fixed_baseline, fixed_amps, phase = args
    #target_dict, stimulus_dict, mu_taus, initial_mu_baseline, initial_mu, sigma_taus, initial_sigma_baseline, initial_sigma, sigma_scale, mu_scale, loss, phase = args

    # Initialize model
    if phase == 0: #fit mu params with fixed sigma params
        """
        print("In Objective function")
        print("x= "+str(x))
        print("fixed_baseline= "+str(fixed_baseline))
        print("fixed_amps= "+str(fixed_amps))
        """
        new_x = np.append(x, [fixed_baseline]) #add fixed sigma params
        new_x = np.append(new_x, fixed_amps)
        #mu_baseline = x[0]
        #mu_amps = x[1:4]
        #print("new_x= "+str(new_x))
        model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus))
        #mu_kernel = ExponentialKernel(mu_taus, mu_amps)
        #model = DetSRP(*(mu_kernel, mu_taus))
        """
        print("*_convert_fitting_params_det(new_x, mu_taus,):")
        print(*_convert_fitting_params_det(new_x, mu_taus, sigma_taus))
        print("*_convert_fitting_params(new_x, mu_taus, sigma_taus)")
        print(*_convert_fitting_params(new_x, mu_taus, sigma_taus))
        """
    elif phase == 1: #fit sigma params with fixed mu params
        #new_x = fixed_baseline + fixed_amps + x
        new_x = np.append([fixed_baseline], fixed_amps) #add fixed sigma params
        new_x = np.append(new_x, x)
        model = ExpSRP(*_convert_fitting_params(new_x, mu_taus, sigma_taus))
    else:
        print("Error, undefined phase (ex: fixed mu params, fixed sigma params)")
    # compute estimates
    mean_dict = {}
    sigma_dict = {}
    for key, ISIvec in stimulus_dict.items():
        mean_dict[key], sigma_dict[key], _ = model.run_ISIvec(ISIvec)

    """
    print("target_dict: ")
    print(target_dict)
    
    print("mean dict: ")
    print(mean_dict)
    """

    # return loss
    if loss == "default":
        if phase == 0:
            return mse_loss(target_dict, mean_dict)
            #return _total_loss(target_dict, mean_dict, sigma_dict)
        else:
            return mse_loss(target_dict, mean_dict)
            #return _total_loss(target_dict, mean_dict, sigma_dict)

    elif loss == "equal":
        return _total_loss_equal_protocol_weights(target_dict, mean_dict, sigma_dict)

    elif callable(loss):
        return loss(target_dict, mean_dict, sigma_dict)

    else:
        raise ValueError(
            "Invalid loss function. Check the documentation for valid loss values"
        )

#--------------------------------------------------------------------

def _total_loss(target_dict, mean_dict, sigma_dict):
    """
    :param target_dict: dictionary mapping stimulation protocol keys to response amplitude matrices
    :param estimates_dict: dictionary mapping stimulation protocol keys to estimated responses
    :return: total nll across all stimulus protocols
    """
    loss = 0
    for key in target_dict.keys():
        loss += _nll(target_dict[key], mean_dict[key], sigma_dict[key])
        #print("target = "+str(target_dict[key]))
        #print("mean = "+str(mean_dict[key]))
    print("loss ="+str(loss))
    return loss

#--------------------------------------------------------------------

def fit_srp_model(
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    initial_mu_baseline = [0],
    initial_mu=[0.01,0.01,0.01], #default  [0.1,0.1, 0.1]
    initial_sigma_baseline=[-1.8],
    initial_sigma=[0.1,0.1,0.1],
    sigma_scale=[4],
    mu_scale=None,
    bounds="default",
    loss="default",
    algo="BFGS" ,
    **kwargs
):
    
    #default: algo="L-BFGS-B" but fix bounds 
    """
    Fitting the SRP model to data using scipy.optimize.minimize
    :param initial_guess: list of parameters:
            [mu_baseline, *mu_amps,sigma_baseline, *sigma_amps, sigma_scale]
    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param mu_taus: predefined time constants for mean kernel
    :param sigma_taus: predefined time constants for sigma kernel
    :param mu_scale: mean scale, defaults to None for normalized data
    :param bounds: bounds for parameters
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param algo: Algorithm for fitting procedure
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize
    """

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)  
        
    #target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fixed_baseline, fixed_amps, phase = args
        
    #agrs before: args=(target_dict, stimulus_dict, mu_taus, initial_mu_baseline, initial_mu, sigma_taus, initial_sigma_baseline, initial_sigma, sigma_scale, mu_scale, loss, true),
    x0=initial_mu_baseline+initial_mu
    
    #select mu params while holding sigmas fixed
    """
    optimizer_res = minimize(
        _objective_function,
        x0=x0,
        method=algo,
        bounds=bounds[0:4],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma, 0),
        **kwargs
    )
    """
    

    optimizer_res = shgo(
        _objective_function,
        bounds=bounds[0:4],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma, 0),
        iters=1,
        **kwargs
    )
    
    mse = optimizer_res.fun
    SD = math.pow(mse, 0.5)
    
    params = _convert_fitting_params(list(optimizer_res.x)+initial_sigma_baseline+initial_sigma, mu_taus, sigma_taus, mu_scale)


    fitted_mu_baseline = params[0]
    fitted_mu_amps = params[1]
    
    #select sigma params while holding mus fixed
    #removed: bounds=bounds,
    x0 = initial_sigma_baseline + initial_sigma
    """
    final_optimizer_res = minimize(
        _objective_function,
        x0=x0,
        method=algo,
        bounds=bounds[4:8],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fitted_mu_baseline, fitted_mu_amps, 1),
        **kwargs
    )
    """
    """
    final_optimizer_res = shgo(
        _objective_function,
        bounds=bounds[4:],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fitted_mu_baseline, fitted_mu_amps, 1),
        **kwargs
    )
    """

    out_x = np.append(fitted_mu_baseline, fitted_mu_amps)
    output = (fitted_mu_baseline, fitted_mu_amps, mu_taus, SD, mu_scale)
    #out_x = np.append(out_x, final_optimizer_res["x"])
    #final_params = _convert_fitting_params(out_x, mu_taus, sigma_taus, mu_scale)
    
    return output, optimizer_res
    #return final_params, optimizer_res

#--------------------------------------------------------------------

def fit_srp_model_v2(
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    initial_mu_baseline = [0.1],
    initial_mu=[0.1,0.1,0.1],
    initial_sigma_baseline=[-1.8],
    initial_sigma=[0.1,0.1,0.1],
    sigma_scale=[0.1],
    mu_scale=None,
    bounds="default",
    loss="default",
    algo="BFGS" ,
    **kwargs
):
    
    #default: algo="L-BFGS-B" but fix bounds 
    """
    Fitting the SRP model to data using scipy.optimize.minimize
    :param initial_guess: list of parameters:
            [mu_baseline, *mu_amps,sigma_baseline, *sigma_amps, sigma_scale]
    :param stimulus_dict: mapping of protocol keys to isi stimulation vectors
    :param target_dict: mapping of protocol keys to response matrices
    :param mu_taus: predefined time constants for mean kernel
    :param sigma_taus: predefined time constants for sigma kernel
    :param mu_scale: mean scale, defaults to None for normalized data
    :param bounds: bounds for parameters
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param algo: Algorithm for fitting procedure
    :param kwargs: keyword args to be passed to scipy.optimize.brute
    :return: output of scipy.minimize
    """

    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)  
        
    #target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fixed_baseline, fixed_amps, phase = args
        
    #agrs before: args=(target_dict, stimulus_dict, mu_taus, initial_mu_baseline, initial_mu, sigma_taus, initial_sigma_baseline, initial_sigma, sigma_scale, mu_scale, loss, true),
    x0=initial_mu_baseline+initial_mu
    
    #select mu params while holding sigmas fixed
    """
    optimizer_res = minimize(
        _objective_function,
        x0=x0,
        method=algo,
        bounds=bounds[0:4],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma, 0),
        **kwargs
    )
    """
    
    optimizer_res = shgo(
        _objective_function,
        bounds=bounds[0:4],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, initial_sigma_baseline, initial_sigma, 0),
        **kwargs
    )
    
    params = _convert_fitting_params(list(optimizer_res.x)+initial_sigma_baseline+initial_sigma, mu_taus, sigma_taus, mu_scale)

    fitted_mu_baseline = params[0]
    fitted_mu_amps = params[1]
    
    #select sigma params while holding mus fixed
    #removed: bounds=bounds,
    x0 = initial_sigma_baseline + initial_sigma
    """
    final_optimizer_res = minimize(
        _objective_function,
        x0=x0,
        method=algo,
        bounds=bounds[4:8],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fitted_mu_baseline, fitted_mu_amps, 1),
        **kwargs
    )
    """
    #"""
    final_optimizer_res = shgo(
        _objective_function,
        bounds=bounds[4:8],
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss, fitted_mu_baseline, fitted_mu_amps, 1),
        **kwargs
    )
    #"""
    
    

    out_x = np.append(fitted_mu_baseline, fitted_mu_amps)
    out_x = np.append(out_x, final_optimizer_res["x"])
    final_params = _convert_fitting_params(out_x, mu_taus, sigma_taus, mu_scale)
    
    
    return final_params, optimizer_res

#--------------------------------------------------------------------

def fit_srp_model_gridsearch(
    stimulus_dict,
    target_dict,
    mu_taus,
    sigma_taus,
    param_ranges="default",
    mu_scale=None,
    sigma_scale=1,
    bounds="default",
    method="TNC",
    loss="default",
    workers=1,
    **kwargs
):
    """
    Fitting the SRP model using a gridsearch.
    :param stimulus_dict: dictionary of protocol key - isivec mapping
    :param target_dict: dictionary of protocol key - target amplitudes
    :param mu_taus: mu time constants
    :param sigma_taus: sigma time constants
    :param target_dict: dictionary of protocol key - target amplitudes
    :param param_ranges: Optional - ranges of parameters in form of a tuple of slice objects
    :param mu_scale: mu scale (defaults to None for normalized data)
    :param sigma_scale: sigma scale in case param_ranges only covers 2 dimensions
    :param bounds: bounds for parameters to be passed to minimizer function
    :param method: algorithm for minimizer function
    :param loss: type of loss to be used. One of:
            'default':  Sum of squared error across all observations
            'equal':    Assign equal weight to each stimulation protocol instead of each observation.
                        This computes the mean squared error for each protocol separately.
    :param workers: number of processors
    """

    # 1. SET PARAMETER BOUNDS
    mu_taus = np.atleast_1d(mu_taus)
    sigma_taus = np.atleast_1d(sigma_taus)

    if bounds == "default":
        bounds = _default_parameter_bounds(mu_taus, sigma_taus)

    # 2. INITIALIZE WRAPPED MINIMIZER FUNCTION
    wrapped_minimizer = MinimizeWrapper(
        _objective_function,
        args=(target_dict, stimulus_dict, mu_taus, sigma_taus, mu_scale, loss),
        bounds=bounds,
        method=method,
        **kwargs
    )

    # 3. MAKE GRID
    if param_ranges == "default":
        param_ranges = _default_parameter_ranges()
    grid = _get_grid(param_ranges)
    starts = _starts_from_grid(grid, mu_taus, sigma_taus, sigma_scale)

    # 4. RUN

    print("STARTING GRID SEARCH FITTING PROCEDURE")
    print("- Using {} cores in parallel".format(workers))
    print("- Iterating over a total of {} initial starts".format(len(grid)))

    print("Make a coffee. This might take a while...")

    # CODE COPIED FROM SCIPY.OPTIMIZE.BRUTE:
    # iterate over input arrays, possibly in parallel
    with MapWrapper(pool=workers) as mapper:
        listres = np.array(list(mapper(wrapped_minimizer, starts)))

    fval = np.array(
        [res["fun"] if res["success"] is True else np.nan for res in listres]
    )

    bestsol_ix = np.nanargmin(fval)
    bestsol = listres[bestsol_ix]
    bestsol["initial_guess"] = starts[bestsol_ix]

    fitted_params = _convert_fitting_params(bestsol["x"], mu_taus, sigma_taus, mu_scale)

    return fitted_params, bestsol, starts, fval, listres

#--------------------------------------------------------------------

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

#--------------------------------------------------------------------

def mse(targets, estimate):
    """
    :param targets: 2D np.array with response amplitudes of shape [n_sweep, n_stimulus]
    :param estimate: 1D np.array with estimated response amplitudes of shape [n_stimulus]
    :return: mean squared errors
    """
    return np.nansum((targets - estimate) ** 2) / np.count_nonzero(~np.isnan(targets))

#--------------------------------------------------------------------

def mse_total_equal_protocol_weights(target_dict, estimates_dict):

    n_protocols = len(target_dict.keys())
    loss_total = 0
    for key in target_dict.keys():
        loss_total += mse(target_dict[key], estimates_dict[key]) * 1 / n_protocols

    return loss_total

#--------------------------------------------------------------------
#plot_mufit for fig8C (edits required)
markersize = 3
capsize = 2
lw = 1
def plot_mufit(axis, target_dict_20, srp_mean):

    xax = np.arange(8)
    print("testing arrays from fit: nanmean, nanstd for 20hz")
    #print(np.nanmean(target_dict_20, 0))
    #print(np.nanstd(target_dict_20, 0))

    #"""  
    axis.errorbar(
        xax,
        np.nanmean(target_dict_20, 0),
        yerr=stat.sem(target_dict["20"], 0, nan_policy='omit'),
        #yerr=np.nanstd(target_dict_20, 0) / 2,
        color="black",
        ls="dashed",
        marker="o",
        label="20 Hz",
        capsize=capsize,
        elinewidth=0.7,
        lw=lw,
        markersize=markersize,
    )

    axis.plot(srp_mean["20"], color=color["srp"], ls="dashed", zorder=10)
    #axis.plot(srp_mean["50"], color=color["srp"], zorder=10)

    axis.set_ylabel(r"norm. EPSC")
    axis.set_xlabel("spike nr.")
    axis.set_ylim(0, 8)
    #axis.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 50, 100])
    #axis.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
    
    axis.legend(frameon=False)


#--------------------------------------------------------------------
# ISI vectors
"""
stimulus_dict = {
    "20": [0] + [50] * 7,
    "100": [0] + [10] * 7
}
"""
stimulus_dict = {
    "20": [0] + [50] * 7,
    #"50": [0] + [20] * 7
}
#print(stimulus_dict)

#replace with extraction code from below and reformat as needed
# Response amplitudes

max_threshold = 0.01
#pickle_file = open("Extracted_Ex_Norm_Mouse_Test.p", "rb")
#pickle_file = open("Extracted_ex_Mouse_Type_Pair_Stim.p", "rb") #main rodent
#pickle_file = open("Extracted_ex_2mM_Mouse_Type_Pair_Stim.p", "rb")
#pickle_file = open("Measures_ex_1.3mM_Human_Type_Pair_Stim.p", "rb") 

measures_name = sys.argv[1]
pickle_file = open(measures_name, "rb")

#pickle_file = open("Extracted_ex_1.3mM_Human_Type_Pair_Stim.p", "rb") #main version for human

recordings = pickle.load(pickle_file)
desired_post = ['nr5a1']
desired_post = ['sst']
#print(recordings)
#unneeded_pre_types = ['nr5a1', 'unknown'] 
for type_pair in recordings.keys():        
    type1, type2 = type_pair
    #extra conditions to target procedure of light on time, don't use these conditions for full run
    
    #print(type_pair)
    chosen_dict = recordings[type_pair] #should be sst
    #print(chosen_dict)
    
    #mean_first = np.median(target_arr_100[:, 0] )
    #attempt to clean data by removing unreasonably low first values
        
    #target_arr_20 = np.asarray(target_list_20)
    #target_arr_100 = np.asarray(target_list_100)
    #trim these to first 8
    #print(chosen_dict[('ic', 20.0, 0.253)])
    target_dict = {}
    training_stim_dict = {} #put this back into use on lines 289 and 290 for version 2
    #print(chosen_dict.keys())
    #print(target_arr_20)
    #stuff below is from version 2
    
    #print("About to print keys")
    #print(chosen_dict.keys())
    for pair_id in chosen_dict.keys():
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
            #print(protocol)
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
                            valid_row = True 
                            for n in range(0, len(added_row)):
                                """
                                if chosen_dict[key][i, n] < 1E-9:
                                    safe_row = False
                                """
                                if chosen_dict[pair_id][protocol][i, n] < 1E-9:
                                    added_row[n] = 1E-9
                                if chosen_dict[pair_id][protocol][i, n] > max_threshold:
                                    valid_row = False
                            if not valid_row:
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
    
    
    mu_kernel_taus = [15, 200, 300]  #orginally [15, 200, 650] for both mu an sigma #this setting for stable supervised
    #mu_kernel_taus = [15, 50, 100, 300]
    sigma_kernel_taus = [15, 100, 300] #orginally [15, 200, 650] for both mu an sigma

    fitted_params = {}
    
    #print(target_dict)
    #print(training_stim_dict)
    for pair_id in target_dict.keys():
        target_all = target_dict[pair_id]
        target_clean = {}
        #print(target_dict)
        for key in target_dict[pair_id]:
            if key != 'pair_ID':
                target_clean[key] = target_dict[pair_id][key]
        #print(target_dict)
        #print(target_dict[pair_data])
        stim_dict = training_stim_dict[pair_id]
        #srp_params, bestfit, _, _, _ = fitting_srp_model(stim_dict, target_clean)
        
        #try but except failed fits (nan, infinite values, failure to converge)
        try:
            srp_params, optimizer_res = fit_srp_model(stim_dict, target_clean, mu_kernel_taus, sigma_kernel_taus)
            fitted_params[pair_id] = srp_params
            #save
            #name = "mass_fits3/srp_fit_full_"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            #name = "mass_fits_two_step2mM/srp_fit_full_"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            #name = "mass_fits_two_stepv5/srp_fit_full_"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            #new testing output dir: testing_fits_rodent_jan23
            folder_name = sys.argv[2]
            name = folder_name+"/"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            #name = "mass_fits_two_step1.3mMHuman/srp_fit_full_"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            #name = "mass_fits4_2mM/srp_fit_full_"+str(type1)+"_"+str(type2)+"_"+str(pair_id)+".p"
            pickle.dump(srp_params, open(name, "wb"))
           
        except:
            print("failed fit for: "+str(type1)+"_"+str(type2)+"_"+str(pair_id))
        
    