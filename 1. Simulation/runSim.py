# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:43:16 2018

@author: jbhud
"""
import sys
sys.path.insert(0, "C:/Users/jbhud/Documents/Thesis/Thesis Code/1. Simulation")
import simulator_class
import time
import os
import pickle

#function for simulating a cohort of n patients
def simpatients(n):
     patients = []
    
     for i in range(n):
        #print("# "+str(i))
        #start = time.time()
        patients = patients + [simulator_class.simulator()]
#       if i%100 == 0:
#           print(str(i))
         #print(str(i) +" "+str(time.time() - start))
        
     return patients

#check and time
start = time.time()
cohort = simpatients(1000)
print(str(time.time()-start))

#run simulation in batches of 10,000 and save each AND in a seperate file, just the states
#homedir = "/media/joshu107/SAMSUNG/" #laptop
#homedir = "F:/" #LIU PC

homedir = "C:/Users/jbhud/Documents/Thesis/Thesis Code/1. Simulation/" #for github

checkpoint = len(os.listdir(homedir+"sim/objects"))

for i in range(checkpoint, 100, 1):
    start = time.time()
    cohort10000 = []
    states10000 = []
    for j in range(10000):
        patient = simulator_class.simulator()
        cohort10000 = cohort10000+[patient]
        states10000 = states10000+[patient.statenums]
    
    with open(homedir+"sim/objects/cohort_"+str(i)+".pkl", "wb") as f:
        pickle.dump(cohort10000, f)
    with open(homedir+"sim/TSonly/states_"+str(i)+".pkl", "wb") as f:
        pickle.dump(states10000, f)
    print(str(time.time()-start))