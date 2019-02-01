# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:40:23 2018

@author: Joshua
"""

# code for changing pandas to dictionaries for thinlinc jupyter

import pickle

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

#cancer probs
with open(home+"1. Simulation/cancerprobs.pkl", "rb")as f:
    cancerprobs = pickle.load(f)
   
pCancer = {str(i):float(cancerprobs.loc[str(i)]/100) for i in range(35, 90)}

with open(home+"1. Simulation/pCancer.pkl", "wb") as f:
    pickle.dump(pCancer, f)
       
    
with open(home+"1. Simulation/deathprobs.pkl", "rb")as f:
    deathprobs = pickle.load(f)

pDeath = {str(i):float(deathprobs.loc[str(i)]) for i in range(35, 90)}   

with open(home+"1. Simulation/pDeath.pkl", "wb") as f:
    pickle.dump(pDeath, f)
    
    
with open(home+"1. Simulation/cancerdeathtimeprobs.pkl", "rb") as f:
    cdf = pickle.load(f)
CDF = {"years":list(cdf["years"]), "probs":list(cdf["probs"])}   
with open(home+"1. Simulation/CDF.pkl", "wb") as f:
    pickle.dump(CDF, f)