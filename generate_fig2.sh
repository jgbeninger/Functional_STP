#!/bin/bash
#This script runs all the code neccessary to generate the plots for figure 2
mkdir fig2
python fig2.py srp_fits_rodent_1.3mM/*
mkdir saved_kernels
mkdir acc_pickles
mkdir single_kernels
python fig2p2.py srp_fits_rodent_1.3mM/*
python process_files_rodent.py srp_fits_rodent_1.3mM/*
python process_files_human.py srp_fits_human/*
mkdir fig3 #create a folder for fig3 because the silhouette plots code generates for both
python silhouette_plots.py
