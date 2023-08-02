#!/bin/bash

#This script calls the relevant Python scripts to load the Allen Institute
#Synaptic Physiology Dataset, extract relevant features for each category
#of our analysis, and then fit our model to synapses in each category.
#Note that this may take a while and you should have at least 190gb of
#available storage and a working internet connection. As an alternative, we 
#also include the output of these scripts in the package. 

python dataset_download.py
python extract_dynamics_rodent.py
python extract_dynamics_rodent_2mM.py
python extract_dynamics_human.py

python extract_STP_Rodent.py
python extract_STP_Rodent_2mM.py
python extract_STP_Human.py

#create folders for SRP fits
mkdir srp_fits_human
mkdir srp_fits_rodent_1.3mM
mkdir srp_fits_rodent_2mM

#fit SRP synapses
python mass_fit_srp2.py Extracted_STP_1.3mM_Human.p srp_fits_human
python mass_fit_srp2.py Extracted_STP_1.3mM_Rodent.p srp_fits_rodent_1.3mM
python mass_fit_srp2.py Extracted_STP_2mM_Rodent.p srp_fits_rodent_2mM

#remove fits rodent fits that include cells with unknown types
#note: this can be changed to keep and analyze those synapses
rm srp_fits_rodent_1.3mM/*unknown*
rm srp_fits_rodent_2mM/*unknown*
