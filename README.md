# Masters Thesis: A Partially Observable Markov Decision Process for Breast Cancer Screening.

In the US, breast cancer is one of the most common forms of cancer and the most lethal.
There are many decisions that must be made by the doctor and/or the patient when dealing
with a potential breast cancer. Many of these decisions are made under uncertainty,
whether it is the uncertainty related to the progression of the patient’s health, or that related
to the accuracy of the doctor’s tests. Each possible action under consideration can
have positive effects, such as a surgery successfully removing a tumour, and negative effects:
a post-surgery infection for example. The human mind simply cannot take into account
all the variables involved and possible outcomes when making these decisions. In
this report, a detailed Partially Observable Markov Decision Process (POMDP) for breast
cancer screening decisions is presented. It includes 151 states, covering 144 different cancer
states, and 2 competing screening methods. The necessary parameters were first set up
using relevant medical literature and a patient history simulator. Then the POMDP was
solved optimally for an infinite horizon, using the Perseus algorithm.

This repository contains all the code used through the Masters Thesis. The script files were divided in five parts, following the order in which they are intended to be executed:
1. Simulation: 
* downloading relevant cancer information through web scraping
* simulating 1 million women's breast cancer histories

2. Probability models:\
maximum likelihood estimation of the transition probabilities using the simulated patient histories

3. Rewards: 
* building the state-based reward model
* estimating the post-cancer rewards by applying the state-based reward model to the simulated patient histories
* building the full reward model combining the state and action based models
            
4. POMDP Set-Up:\
formatting all the parameters, creating the .POMDP input file.\
\
Between parts 4 and 5:\
Solve the POMDP using a POMDP solver (Erwin Walraven's PERSEUS implemenation was used, refer to https://www.erwinwalraven.nl/solvepomdp/)

5. Policy Interpretation:\
using the alpha vectors contained in the .alpha output file to form the optimal policy as a function of the belief state
