# Compositional Recombination Relies on a Distributed Cortico-Cerebellar Network
This repository contains the code used in my project *"Compositional Recombination Relies on a Distributed Cortico-Cerebellar Network"* . 

**Code** was written using a combination of MATLAB and Python scripts.

**Data** was downloaded from *OpenNeuro* [ds003701](https://openneuro.org/datasets/ds003701/versions/1.0.1) and originally collected from this [paper](https://www.nature.com/articles/s41467-017-01000-w).
## code
The code folder contains the code required to run the analyses and produce the figures. The following scripts will be described in an order that fits with the manuscript and analyses. Note that these scripts are not functions and should be viewed similar to a notebook. Figures do not appear in manuscript order but in the order that they were made using the data.

- [dataprep_behavioural.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/dataprep_behavioural.m) includes code for processing the data after pre-processing. Reads in the behavioural data (.tsv) and timeseries data (.mat), and removes missing values, as well as separates correct and incorrect trials. Statistical analysis of behavioural differences are also calculated here (refer to manuscript for more details). **The outputs from this script are used for all fMRI analyses described below.** This script has code to produce Figures 1d-f.

- [FIR_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/FIR_analysis.m) creates the design matrix to model the BOLD timeseries using a *Finite Impulse Response (FIR) model*.

- [PLS_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/PLS_analysis.m) runs the *Partial Least Squares (PLS) analysis*. Main analyses included in this script: PLS construction, split-test reliability, permutation testing of latent variables, generating bootstrap samples, separating rule-dependent and rule-independent regions using *Euclidean distance* and *Mean FIR response*. This scrips produces Figures 2g-i, 3a-e, and Supplementary Figure S1.

- [FC_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/FC_analysis.m) calculates functional connectivity matrices during the trial period. The script identifies significant edges between rule-dependent and rule-independent regions per task domain (Motor, Logic, Sensory), runs a conjunctive analysis, identifies Recombination regions, and checks the stability of the regions using Bootstrap Ratios (BSR). The script produces Figure 3g and Supplementary Figure S3.

- [dimensionality_analysis.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/dimensionality_analysis.m) calculates the dimensionality measures (participation ratio, participation coefficient or integration, response similarity across mini-blocks) in the fMRI data. This script produces Figure 4a-c and Supplementary Figure S2.

- [enhanced_activity_generator.py](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/enhanced_activity_generator.py) simulates unit activity using the pretrained models from [Yang et al., 2019](https://www.nature.com/articles/s41593-018-0310-2). Models and original code to train the models are available on the [original github](https://github.com/gyyang/multitask). Note that the original models were built using Python 2.7 and 3.6 and require these environments to train the models. Alternatively, you can use the [environment](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/environment.yaml) that we used to simulate the models after downloading.

- [RNN_analysis.py](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/RNN_analysis.py) contains code to explore the Yang networks and produce the Task Variance measures originally used in [Yang et al., 2019](https://www.nature.com/articles/s41593-018-0310-2). Models and original code to train the models are available on the [original github](https://github.com/gyyang/multitask).

- [RNN_variance.m](https://github.com/ShineLabUSYD/Compositionality_CPRO/blob/main/code/RNN_variance.m) includes sanity checks that our simulated unit activity is similar to the activity simulated in the original paper. This script also calculates the dimensionality measures (participation ratio, participation coefficient, response similarity across tasks) in the RNN models. This scrips produces Figures 4d-f. 
