# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:02:09 2018

@author: jbhud
"""

#execute policy
import pickle
import numpy as np
import matplotlib.pyplot as plt

home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"
#load patients
with open(home+"5. Policy Interpretation/testpatients.pkl", "rb") as f:
    cohort = pickle.load(f)
#convert to 6month time step 
temp = cohort
for i in range(len(cohort)):
    ts = cohort[i]
    temp[i] = [ts[t] for t in range(len(ts)) if t%6 == 0]
cohort = temp
#load matrices
with open(home+"4. POMDP Set-Up/initProbs.pkl", "rb") as f:
    initProbs = pickle.load(f)
#initProbs = np.array([1/148]*148+[0, 0, 0])   #problem: more cancer states so not really uniform
#initProbs = np.array([1/37]+[0]*3+[1/37]*36+[0]*111)
with open(home+"4. POMDP Set-Up/transProbs.pkl", "rb") as f:
    transProbs = pickle.load(f)

with open(home+"4. POMDP Set-Up/obsProbs.pkl", "rb") as f:
    obsProbs = pickle.load(f)


with open(home+"4. POMDP Set-Up/rewards.pkl", "rb") as f:
    R = pickle.load(f)

with open(home+"5. Policy Interpretation/policy.pkl", "rb") as f:
    policy = pickle.load(f)

def executePolicy(patient, policy, initProbs, transProbs, obsProbs, Rmodel, manual):
    n = len(initProbs) #number of states
    nodes = policy["nodes"]
    alphas = policy["alphas"]
    horizon = len(patient)
    bel_seq = np.zeros([horizon, n])
    bel_seq[0, ] = initProbs
    R_seq = np.zeros(horizon)
    a_seq = [0]*horizon
    o_seq = [0]*horizon
    
    #set starting defaults for state and action
    optact = "nothing"
    state = 0
    for t in range(horizon-1):
        #determine true state
        if ((optact == "surgery") & (state in set(list(range(4, 150))))) | (state == 149):
            state = 149
            a_seq[t] = "nothing"
            o_seq[t] = "positive-test"
            bel_seq[t] = np.array([0]*148+[1, 0, 0])
        else:
            state = patient[t]
            #take best action based on dot product of alphas and current belief state
            VF = np.dot(alphas, bel_seq[t, ])
            optact = nodes[np.argmax(VF)]
            if (manual == "true") & (optact == "surgery"):
                VF = np.delete(VF, np.argmax(VF))
                optact = nodes[np.argmax(VF)]
            #record reward
            R_seq[t] = Rmodel[optact][state]
            #select appropriate trans and obs matrices
            T = transProbs[optact]
            O = obsProbs[optact]
            #draw observation at random using observation probs for the current state
            o = np.random.choice([0, 1, 2], p = list(O[state, ]))
            #update belief
            oldbelief = bel_seq[t, ]
            newbelief = oldbelief
            for s2 in range(n):
                Txb = np.zeros(n) #transprob times belief for each state
                for s1 in range(n):
                    Txb[s1] = T[s2, s1]*oldbelief[s1]
                newbelief[s2] = O[state, o]*np.sum(Txb)
            #normalise
            newbelief = newbelief/np.sum(newbelief)
            #keep track of action taken and observation incurred
            a_seq[t] = optact
            o_seq[t] = ["positive-test", "negative-test", "dead"][o]
            bel_seq[t+1] = newbelief
            
    totalR = np.sum(R_seq)
    ts = np.concatenate((np.transpose(np.array([patient])), np.transpose(np.array([a_seq])), np.transpose(np.array([o_seq])), np.transpose(np.argmax(bel_seq, 1))), axis = 1)
    return{"timeseries":ts, "totalreward":totalR}
    
    ###UNFINISHED: TURN OUTPUT INTO PANDA!!!
    
    
#test for different patient scenarios
#cancer patient
plt.plot(cohort[0])
res1 = executePolicy(cohort[0], policy, initProbs, transProbs, obsProbs, R, manual = "false")
res2 = executePolicy(cohort[0], policy, initProbs, transProbs, obsProbs, R, manual = "true")
#plot to find 1 for each different patient type
n = 3
plotdata=cohort[0:n]
t = [35+i/2 for i in range(110)]
for i in range(n):
    plt.plot(t, plotdata[i], label=str(i))
#plt.ylim((0, 151))
plt.legend(loc="upper left")
plt.xlabel("Age")
plt.ylabel("State")
plt.show()
#healthy life
plt.plot(t, cohort[18])
res = executePolicy(cohort[18], policy, initProbs, transProbs, obsProbs, R)
res["totalreward"]
#cancer patient
plt.plot(cohort[0])
res = executePolicy(cohort[0], policy, initProbs, transProbs, obsProbs, R)
#benign tumour patient
