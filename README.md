# Compositional Recombination Relies on a Distributed Cortico-Cerebellar Network
This repository contains the code used in my project *"Compositional Recombination Relies on a Distributed Cortico-Cerebellar Network"* . 

**Code** was written using a combination of MATLAB and Python scripts.

**Data** was downloaded from *OpenNeuro* [ds003701](https://openneuro.org/datasets/ds003701/versions/1.0.1) and originally collected from this [paper](https://www.nature.com/articles/s41467-017-01000-w).
## code
The code folder contains the code required to run the analyses and produce the figures. The following scripts will be described in an order that fits with the manuscript and analyses. Note that these scripts are not functions and should be viewed similar to a notebook.

- [dataprep_behavioural.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/dataprep_behavioural.m) includes code for processing the data after pre-processing. Reads in the behavioural data (.tsv) and timeseries data (.mat), and removes missing values, as well as separates correct and incorrect trials. Statistical analysis of behavioural differences are also calculated here (refer to manuscript for more details). **The outputs from this script are used for all analyses described below.** This script has code to produce Figures 1d-f.
- [FIR_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/FIR_analysis.m) creates the design matrix to model the BOLD timeseries using a *Finite Impulse Response (FIR) model*.
- [PLS_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/PLS_analysis.m) runs the *Partial Least Squares (PLS) analysis*. Main analyses included in this script: PLS construction, split-test reliability, permutation testing of latent variables, generating bootstrap samples, separating rule-dependent and rule-independent regions using *Euclidean distance* and *Mean FIR response*. This scrips produces Figures 2g-i, 3a-e.
- [nrgCalc_WM.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/nrgCalc_WM.m) runs *energy landscape analysis* ([Munn et al., 2021](https://www-nature-com.ezproxy.library.sydney.edu.au/articles/s41467-021-26268-x)). Original energy landscape code can be found [here](https://github.com/ShineLabUSYD/Brainstem_DTI_Attractor_Paper). This script also produces Figures 4f-h.
- [workingmemory_1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/workingmemory_1000.m), [FIR_analysis1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/FIR_analysis1000.m), and [ldaMATLAB1000.m](https://github.com/JoshuaBTan/WM_Manipulation/blob/main/Code/ldaMATLAB1000.m) contains similar code to the main analyses but replicated using 1000 Schaefer cortical 
