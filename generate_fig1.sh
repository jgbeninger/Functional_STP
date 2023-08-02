#!/bin/bash
#This program runs the neccessary scripts to generate the subfigures of 
#figure 1 and store them in a folder titled 'fig1'
mkdir fig1
mkdir fig1/indfits
python fig1_mse_plots.py srp_fits_rodent_1.3mM/*
python fig1_individual_pairs.py srp_fits_rodent_1.3mM/*

