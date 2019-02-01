# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:54:12 2018

@author: jbhud
"""

import numpy as np
import pickle

nodes = []

with open(r"C:\Users\jbhud\Documents\Thesis\Thesis Code\5. Policy Interpretation\breastcancerNEWNEW.alpha") as f:
    lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    alphas = np.zeros([int(len(lines)/3), 151])
    for i in range(len(lines)):
        if i%3 == 0:
            nodes = nodes+[lines[i]]
            alphas[int(i/3), ] = [float(j) for j in lines[i+1].split()]
            
#find best action for each state assuming we are 100% certain on being in that state
idx = np.argmax(alphas, 0)
optact_certain = [nodes[i] for i in idx]


#save as text for spreadsheet input
with open(r"C:\Users\jbhud\Documents\Thesis\Thesis Code\5. Policy Interpretation\optact.txt", "w") as f:
    for item in optact_certain:
        f.write(item+"\n")
        
#save policy as pickle
policy = {"nodes":nodes, "alphas":alphas}
with open("C:/Users/jbhud/Documents/Thesis/Thesis Code/5. Policy Interpretation/policy.pkl", "wb") as f:
    pickle.dump(policy, f)


#proper way to get optimal action given a belief vector
def optAction(policy, belief):
    nodes = policy["nodes"]
    alphas = policy["alphas"]
    return nodes[np.argmax(np.dot(alphas, belief))] 
          
#try different belief vectors
#b_init = np.array([1]+[0]*150) 
#max(np.dot(alphas, b_init))


#look at cancer stages, assuming certainty
with open(r"C:/Users/jbhud/Documents/Thesis/Thesis Code/5. Policy Interpretation/state2stage.txt", "r") as f:
    lines = f.readlines()
stages = [line.rstrip() for line in lines]
b_stages = [0]*4
b_stageI = np.array([1 if stage=="1" else 0 for stage in stages])
b_stages[0] = b_stageI/np.sum(b_stageI)
b_stageII = np.array([1 if stage=="2" else 0 for stage in stages])
b_stages[1] = b_stageII/np.sum(b_stageII)
b_stageIII = np.array([1 if stage=="3" else 0 for stage in stages])
b_stages[2] = b_stageIII/np.sum(b_stageIII)
b_stageIV = np.array([1 if stage=="4" else 0 for stage in stages])
b_stages[3] = b_stageIV/np.sum(b_stageIV)

optA_stages = [optAction(policy, bel) for bel in b_stages]

#healthy vs cancer
b_cancer = [0]*11
percent = np.array(range(0, 110, 10))/100 #percentage certainty of having cancer vs being healthy
for i in range(11):
    b_cancer[i] = np.array([(1-percent[i])/4]*4+[(percent[i])/144]*144+[0]*3)
    print(np.sum(b_cancer[i]))
optA_HvC = [optAction(policy, bel) for bel in b_cancer]

b_healthy = np.array([1/4]*4+[0]*147)
b_cancer100 = np.array([0]*4+[1/144]*144+[0]*3)