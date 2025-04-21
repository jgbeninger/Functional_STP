Copyright (C) 2023 John Beninger, Julian Rossbroich, Richard Naud

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.


# Functional_STP
Code repository for the paper Functional Subtypes of Synaptic Dynamics in Mouse and Human
Final Version:
(see: https://www.cell.com/cell-reports/fulltext/S2211-1247(24)00113-X?uuid=uuid%3A545e8120-bfa9-4120-82ce-8941a6ade19f )
Early Version:

(see: https://www.biorxiv.org/content/10.1101/2023.05.23.541971v1.abstract).
To reproduce the paper's main figure run the following bash scripts in sequence:
run_fits.sh
generate_fig1.sh
generate_fig2.sh
generate_fig3.sh
generate_fig4.sh

These scripts reproduce the paper with figure numbering corresponding to the BioRxiv version of the manuscript, linked above.  

The remaining scripts in the repository support our testing of continuum hypotheses in clustering as well as additional analyses not included in the main figures of the paper, many of which are summarized in supplemental figures.

# Tutorial
We have produced a streamlined and user friendly introductory tutorial paper that provides support for implementing some of the analysis contained in this paper and the original SRP model paper. The paper is titled "Computational protocol for modeling and analyzing synaptic dynamics using SRPlasticity" and can be found in the journal Star Protocols.

Further, a companion Jupyter notebook containing the tutorial material has been integrated with the SRP model repository: https://github.com/nauralcodinglab/srplasticity 

At the time of writing, the paper can be found here: https://www.sciencedirect.com/science/article/pii/S2666166725000589

# Archived Version
An archived version of this codebase has been generated to correspond to the published version of this paper: https://zenodo.org/doi/10.5281/zenodo.10542491. We will likely also maintain a live public version of the code on Github: https://github.com/jgbeninger/Functional_STP 

## Paper Citations
We ask all users of this codebase to cite our paper: 

@article{beninger2024functional,
  title={Functional subtypes of synaptic dynamics in mouse and human},
  author={Beninger, John and Rossbroich, Julian and Toth, Katalin and Naud, Richard},
  journal={Cell Reports},
  volume={43},
  number={2},
  year={2024},
  publisher={Elsevier}
}


We also ask users of the SRP model to cite the original paper on the model:

@article{rossbroich2021linear,
  title={Linear-nonlinear cascades capture synaptic dynamics},
  author={Rossbroich, Julian and Trotter, Daniel and Beninger, John and T{\'o}th, Katalin and Naud, Richard},
  journal={PLoS computational biology},
  volume={17},
  number={3},
  pages={e1008013},
  year={2021},
  publisher={Public Library of Science San Francisco, CA USA}
}
