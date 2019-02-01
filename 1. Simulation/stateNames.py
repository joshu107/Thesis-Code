# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:54:18 2019

@author: jbhud
"""

#load list of states from before midterm
import pickle
home="C:/Users/jbhud/Documents/Thesis/Thesis Code/"
#
naturalStates = []
with open(home+"1. Simulation/listNatStates.txt", "r") as f:
    for row in f:
        naturalStates = naturalStates + [row.rstrip()]
    
with open(home+"1. Simulation/naturalStates.pkl", "wb") as f:
        pickle.dump(naturalStates, f)

        
stateDict = {naturalStates[i]:i for i in range(len(naturalStates))}
#148 was changed to be the postcancer state, move death states along one
stateDict["1"] = 149
stateDict["2"] = 150

with open(home+"1. Simulation/stateDict.pkl", "wb") as f:
    pickle.dump(stateDict, f)