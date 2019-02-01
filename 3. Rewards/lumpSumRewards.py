# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 13:35:15 2018

@author: jbhud
"""
import pickle
import numpy as np

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

with open(home+"1. Simulation/sim/TSonly/states_0.pkl", "rb") as f:
    cohort = pickle.load(f)
#calculated total expected reward from any state till death (for post treatment lump sum)
def lumpsumR(cohort, Rmodel, discount=0.99):
    countR = np.zeros(150)
    countS0 = np.zeros(150)
    for ts in cohort:
        ts = [ts[t] for t in range(len(ts)) if t%6 == 0]
        rewards = [Rmodel[ts[t]]*discount**t for t in range(len(ts))]
        for t in range(len(ts)):
            countR[ts[t]] += np.sum(rewards[t:])
            countS0[ts[t]] += 1
    return np.array([countR, countS0])
    
Rmodel = np.array([0.5]*148+[0, 0])
Qscores = []
i=0
with open(home+"3. Rewards/hrqol_states.txt", "rb") as f:
    for row in f:
        Qscores = Qscores + [float(row)]
Qscores = Qscores + [0, 0]
Qscores = np.array(Qscores)
Rmodel_QA = Qscores*Rmodel
EL = lumpsumR(cohort, Rmodel_QA)

fullcount = np.zeros([2, 150])
for i in range(100):
    with open(home+"1. Simulation/sim/TSonly/states_"+str(i)+".pkl", "rb") as f:
        cohort = pickle.load(f)
        fullcount = fullcount + lumpsumR(cohort, Rmodel_QA)

with open(home+"3. Rewards/lumpSum_QALY.pkl", "wb") as f:
    pickle.dump(fullcount[0]/fullcount[1], f)

with open(home+"3. Rewards/Rmodel_state_QALY.pkl", "wb") as f:
    pickle.dump(Rmodel_QA, f)
    
ER_posttreat = fullcount[0]/fullcount[1]