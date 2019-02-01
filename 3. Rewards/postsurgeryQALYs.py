# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:10:54 2018

@author: jbhud
"""

#best treatment for each state

import pickle
import numpy as np
import csv

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

with open(home+"3. Rewards/lumpSum_QALY.pkl", "rb") as f:
    lumpsumR = pickle.load(f)

cureR = np.array([lumpsumR[0]]*36+[lumpsumR[1]]*36+[lumpsumR[2]]*36+[lumpsumR[3]]*36)
nocureR = lumpsumR[4:148]
cureprobs = np.zeros([150, 6])
with open(home+"3. Rewards/cureprobs.csv", encoding="utf-8-sig") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    i=0
    for row in csvReader:
        cureprobs[i, ] = row
        i += 1
cureprobs= cureprobs[4:148, :]

#calculate chemo QALY cost (only treatment affecting QALYs)
with open(home+"3. Rewards/Rmodel_state_QALY.pkl", "rb") as f:
    Rmodel_QA = pickle.load(f)
Rmodel_QA = Rmodel_QA[4:148]

chemocost = np.array([-2/3*23.7/100*qaly for qaly in Rmodel_QA])
#calculate Expected Reward of taking each treatment action based on cure probs
#for each CANCER state
ER = np.zeros([144, 6])
for j in range(6):
    if j in set([1, 4, 5]): #chemo treatments
        ER[:, j] = cureprobs[:, j]*cureR+(1-cureprobs[:, j])*nocureR+chemocost
    else:
        ER[:, j] = cureprobs[:, j]*cureR+(1-cureprobs[:, j])*nocureR

optA_ER = np.max(ER, axis = 1) #maximum ER if take optimal action
optA = np.argmax(ER, axis = 1) #action that maximised ER (just 1st if several)
surgerycost = np.array([-44.8/100*qaly for qaly in Rmodel_QA])
surgeryExpQALYs = optA_ER+surgerycost
with open(home+"3. Rewards/postsurgeryRewards.pkl", "wb") as f:
    pickle.dump(optA_ER, f)


