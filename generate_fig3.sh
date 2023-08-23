#!/bin/bash
#This script runs all the code necessary to generate the plots for figure 3
mkdir fig3
python process_files_rodent.py srp_fits_rodent_1.3mM/*
python process_files_human.py srp_fits_human/*
mkdir fig2 #create a folder for fig2 because the Silhouette plots script generates figure for both fig2 and fig3
python silhouette_plots.py
python Human_Single_kernel_plots.py srp_fits_human/*
python joint_pc_projections.py
python human_plota.py srp_fits_human/*
mkdir fig4 #create a folder for fig4 because the similarity measures script generates plots for both fig3 and fig4
python similarity_measures.py
