# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:56:40 2019

@author: jbhud
"""

import pickle
import numpy as np

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

#load matrices
with open(home+"4. POMDP Set-Up/initProbs.pkl", "rb") as f:
    initProbs = pickle.load(f)
    
with open(home+"4. POMDP Set-Up/transProbs.pkl", "rb") as f:
    transProbs = pickle.load(f)

with open(home+"4. POMDP Set-Up/obsProbs.pkl", "rb") as f:
    obsProbs = pickle.load(f)


with open(home+"4. POMDP Set-Up/rewards.pkl", "rb") as f:
    R = pickle.load(f)
 
    #extra param for tweaking biopsy cost
#param = "uniformbel"
#R["surgery"][0:4] = [r*(100-param)/100 for r in R["surgery"][0:4]]
#paramname = str(param).replace(".", "_")
#initProbs = [0.998]+[0]*3+[0.002/72]*36+[0]*111 #uniform belief between cancer and no cancer

(nStates, nObs) = obsProbs["cbe"].shape
actions = list(transProbs.keys())
action_str = " ".join(actions)

discount = 0.99

with open(home+"4. POMDP Set-Up/breastcancer.POMDP", "w") as f:
    #set up
    f.write("discount: "+str(discount)+"\n")
    f.write("values: reward\n")
    f.write("states: "+str(nStates)+"\n")
    f.write("actions: "+action_str+"\n")
    f.write("observations: positive-test negative-test dead\n")
    f.write("\n")
    #init probs
    f.write("start:\n")
    for j in range(nStates):
        f.write(str(initProbs[j])+" ")
    f.write("\n")
    f.write("\n")
    #transition probs
    for action in actions:
        f.write("T:"+action+"\n")
        for row in range(nStates):
            for col in range(nStates):
                f.write(str(round(transProbs[action][row, col], 4))+" ")
            f.write("\n")
        f.write("\n")
    #observation probs
    for action in actions:
        f.write("O:"+action+"\n")
        for row in range(nStates):
            for col in range(nObs):
                f.write(str(obsProbs[action][row, col])+" ")
            f.write("\n")
        f.write("\n")
    f.write("\n")
    #rewards
    for action in actions:
        for i in range(nStates):
            f.write("R : "+action+" : "+str(i)+" : * : * "+str(R[action][i])+"\n")
        f.write("\n")