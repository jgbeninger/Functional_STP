#!/bin/bash
#This script runs all the code necessary to generate the plots for figure 4
mkdir fig4
mkdir fig3 #created folder for fig3 because the similarity_measures script creates plots for both fig3 and fig4
python process_files_rodent.py srp_fits_rodent_1.3mM/*
python process_files_human.py srp_fits_human/*
python similarity_measures.py

mkdir ./fig4/dists
#python bootstrap_fig4b.py srp_fits_rodent_1.3mM/*
python bootstrap_diff1.3mMdownsampledmlp.py srp_fits_rodent_1.3mM/*
#python bootstrap_diff2mMmlp.py srp_fits_rodent_2mM/*
python downsample_comp.py
