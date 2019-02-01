# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 17:08:27 2018

@author: jbhud
"""
### NEW SETUP ### 151 states, 4 actions
import pickle
import csv
import numpy as np

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

#initial model
initProbs = np.array([1]+[0]*150) #certain of healthiness and age
#initProbs = np.array([1/37]+[0, 0, 0]+[1/37]*36+[0]*111) #uniform over alive states under 50
with open(home+"4. POMDP Set-Up/initProbs.pkl", "wb") as f:
    pickle.dump(initProbs, f)
   
#transition model
#load natural transitions
with open(home+"2. Probability Models/Tprobs.pkl", "rb") as f:
     Tprobs = pickle.load(f)

    
transProbs = {}
actions = ["nothing", "cbe", "mammography", "surgery"] #only 4 now
for action in actions[0:3]:
    mat = np.identity(151)
    mat[0:148, 0:148] = Tprobs[0:148, 0:148]
    mat[149:151,0:148] = Tprobs[148:150, 0:148]
    mat[0:148, 149:151] = Tprobs[0:148, 148:150]
    mat[149:151, 149:151] = Tprobs[148:150, 148:150]
    transProbs[action] = mat

#code for checking mat format in txt file
with open(home+"4. POMDP Set-Up/Tprobs.txt", "w") as f:
    for i in range(151):
        for j in range(151):
            f.write(str(mat[i, j])+" ")
        f.write("\n")
        
    
mat = np.identity(151)
mat[4:148, 4:148] = 0
mat[4:148, 148] = 1
transProbs["surgery"] = mat

with open(home+"4. POMDP Set-Up/transProbs.pkl", "wb") as f:
    pickle.dump(transProbs, f)
    
#Observation model
nStates = 151
nObs = 3
Oprobs_mam = np.zeros(shape = [nStates, nObs])
Oprobs_cbe = np.zeros(shape = [nStates, nObs])
with open(home+"2. Probability Models/obsmodel.csv") as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    i = 0
    for row in csvReader:
        Oprobs_mam[i, 0] = row[1]
        Oprobs_mam[i, 1] = round(1 - Oprobs_mam[i, 0], 3)
        Oprobs_cbe[i, 0] = row[2]
        Oprobs_cbe[i, 1] = round(1 - Oprobs_cbe[i, 0], 3)
        i += 1
#adjust death state values
Oprobs_mam[149:151, 1] = 0
Oprobs_mam[149:151, 2] = 1
Oprobs_cbe[149:151, 1] = 0
Oprobs_cbe[149:151, 2] = 1
#postcancer state
Oprobs_cbe[148, 1] = 0
Oprobs_cbe[148, 0] = 1
Oprobs_mam[148, 1] = 0
Oprobs_mam[148, 0] = 1
#surgery has perfect observation
Oprobs_surg = np.zeros(shape = [nStates, nObs])
Oprobs_surg[0:4, 1] = np.array([1]*4) #healthy states
Oprobs_surg[4:149, 0] = np.array([1]*145) #cancer states
Oprobs_surg[149:151, 2] = np.array([1]*2)

Oprobs_nothing = np.array([[0.5, 0.5, 0]]*148+[[1, 0, 0]]+[[0, 0, 1]]*2)
obsProbs = {"nothing":Oprobs_nothing, "cbe":Oprobs_cbe, "mammography":Oprobs_mam, "surgery":Oprobs_surg} 
with open(home+"4. POMDP Set-Up/obsProbs.pkl", "wb") as f:
    pickle.dump(obsProbs, f)

#reward model
    #ACTION NOTHING + CBE
#load state -based QALY reward model, add post cancer state reward of 0
with open(home+"3. Rewards/Rmodel_state_QALY.pkl", "rb") as f:
    R_nothing = pickle.load(f)
    
R_nothing = list(R_nothing)+[0]
    #CBE -> 1 day of stress (same as mammography)
R_cbe = [(1-1/180*9.76/100)*qaly for qaly in R_nothing]
    #ACTION MAMMOGRAPHY
R_mammography = [qaly-1/8*9.76/100*qaly for qaly in R_nothing]
    #ACTION SURGERY
#if cancer
R_surg_true = [(1-5/12*37.9/100-44.8/100)*qaly for qaly in R_nothing]
with open(home+"3. Rewards/postsurgeryRewards.pkl", "rb") as f:
    endR = pickle.load(f)
endR = [0]*4+list(endR)+[0]*3
R_surg_true = np.array(R_surg_true)+np.array(endR)
R_surg_true[0:4] = 0
#if no cancer, assume that pre-surgery diagnostical test catch the false positive, so cost is for those
#set manual cost of misdiagnosis, otherwise system takes surgery action for all healthy states
negreward = 0
R_surg_false = np.array([(1-1/4*37.9/100)*qaly - negreward for qaly in R_nothing])
R_surg_false[4:151] = 0
#ALTERNATIVE give no reward for misdiagnosis, effectively losing 6months of life
#R_surg_false = np.array([0]*151)

R = {"nothing":np.array(R_nothing), "cbe":np.array(R_cbe), "mammography":np.array(R_mammography), "surgery":{"true":R_surg_true, "false":R_surg_false}}

#new format
Rsurg = list(R["surgery"]["false"][0:4])+list(R["surgery"]["true"][4:151]) 
R["surgery"] = np.array(Rsurg)

with open(home+"4. POMDP Set-Up/RewardFull.txt", "w") as f:
    for action in actions:
        f.write(str(action)+" ")
    f.write("\n")
    for i in range(nStates):
        for action in actions:
            f.write(str(R[action][i])+" ")
        f.write("\n")
    f.write("\n")
with open(home+"4. POMDP Set-Up/rewards.pkl", "wb") as f:
    pickle.dump(R, f)
