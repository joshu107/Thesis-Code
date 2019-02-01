# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import time
import numpy as np

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

#code for counting transitions using 10,000 states time series'
with open(home+"1. Simulation/naturalStates.pkl", "rb") as f:
        listofStates = pickle.load(f)
        
nStates = len(listofStates)

def Tcounts(cohort, nStates = 150):
    counts = np.zeros(shape = [nStates, nStates])
    for ts in cohort:
        ts = [ts[i] for i in range(len(ts)) if i%6 == 0]
        if ts[109] == 148:
            ts = ts+[148]
        else:
            ts = ts+[149]
        for (i, j) in zip(ts, ts[1:]):
            counts[i,j] += 1
    return counts

fullcount = np.zeros(shape = [nStates, nStates])
for i in range(100):
    with open(home+"1. Simulation/sim/TSonly/states_"+str(i)+".pkl", "rb") as f:
        cohort = pickle.load(f)
        fullcount = fullcount + Tcounts(cohort)

#check adds up to 110million
np.sum(fullcount) #success!

#see if all states appear
np.sum(fullcount, axis = 1) #no zeros so all states appear

with open(home+"2. Probability Models/Tcounts.pkl", "wb") as f:
    pickle.dump(fullcount, f)
    
    
#turn to probs

with open(home+"2. Probability Models/Tcounts.pkl", "rb") as f:
    Tcounts = pickle.load(f)


n = len(Tcounts)
Tprobs = np.zeros(shape = [n, n])
for i in range(n):
    Tprobs[i, ] = Tcounts[i, ]/np.sum(Tcounts[i, ])

#check if rows add up to 1 !no rounding for now!
for i in range(n):
    #for j in range(n):
       # Tprobs[i, j] = round(Tprobs[i, j]
    print(np.sum(Tprobs[i, ]))

#round
nStates = len(Tprobs)

def matround(mat):
    #function for rounding probs in a trans/obs model, ensuring rows sum to 1 still 
    (n, m) = mat.shape
    for i in range(n):
        for j in range(m):
            mat[i, j] = round(mat[i, j], 3)
        rem = 1-np.sum(mat[i, ])
        if rem > 0:
            mat[i, m-1] += rem #add excess to death all causes
        if rem < 0:
            mat[i, i] += rem  
    return mat
        

Tprobs = matround(Tprobs)
#test
for i in range(nStates):
    print(np.sum(Tprobs[i, ]))
#fail so reappply function
Tprobs = matround(Tprobs)
for i in range(nStates):
    print(np.sum(Tprobs[i, ]))
#1 error cancer death to all cause death transition: manually set
Tprobs[148, 148] = 1
Tprobs[148, 149] = 0
    
with open(home+"2. Probability Models/Tprobs.pkl", "wb") as f:
    pickle.dump(Tprobs, f)
