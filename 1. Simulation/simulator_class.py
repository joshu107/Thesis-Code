# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:25:48 2018

@author: jbhud
"""

import numpy as np
from numpy.random import gamma as rgamma
from numpy.random import binomial as rbinom
from numpy.random import multinomial as rmultinom
from numpy.random import uniform as runif
from numpy.random import randint
from numpy import exp, log, pi
import pickle

            
class simulator(object):
    
    def __init__(self, startAge = 35, endAge = 90, interval = 1, d0=0.2, dmax=8,
                 homedir = "C:/Users/jbhud/Documents/Thesis/Thesis Code/1. Simulation"):
        #### OBJECT FOR SIMULATION OF A SINGLE PATIENT'S CANCER HISTORY/STATUS
        ###Args:
        #startAge, endAge: starting and end ages for simulating patient history between (in years)
        #interval: interval between time steps expressed in months
        #d0: assumed starting size of a tumor (diameter in cm)
        #dmax: assumed maximum size of a tumor, one the growth function will tend to as t->Inf (diameter in cm)
        
        ###Returns:
        #months: time range over which the simulation is done (time series in months)
        
        self.startAge = startAge
        self.endAge = endAge
        self.interval = interval
        self.d0 = d0
        self.dmax = dmax
        
        #load data from external sources
        with open(homedir+"pCancer.pkl", "rb")as f:
             self.pCancer = pickle.load(f)
                
        with open(homedir+"pDeath.pkl", "rb")as f:
            self.pDeath = pickle.load(f)
            
        with open(homedir+"CDF.pkl", "rb") as f:
            self.cdf = pickle.load(f)
            
        with open(homedir+"stateDict.pkl", "rb") as f:
            self.stateDict = pickle.load(f)
            
        #generate age range in months
        self.months = range(self.startAge*12, self.endAge*12, self.interval)
        
        #SIMULATE!
        self.simHistory()
    
    def genAlpha(self):
        #draws a random alpha (initial growth rate from the gamma(1.2, 0.1) distribution
        self.alpha = rgamma(shape = 1.2, scale = 0.1)
        
    def sphereV(self, d):
        #spherical volume function
        return pi/6*d**3

    def tumSize(self, t, a):
        #function for calculating tumor diameter at time t, given a patient's initial growth rate a
        return self.d0*exp(log(self.dmax/self.d0)*(1-exp(-a*t)))

    def tumSizeRate(self, t, a):
        #function for calculating tumor volume growth rate at time t, given a patient's initial growth rate a
        #(CISNET simulator adjusted the diameter by 75% after experiments)
        V0 = self.sphereV(d=0.75*self.d0)
        Vmax = self.sphereV(d=0.75*self.dmax)
        return exp(-a*t)*V0*a*(Vmax/V0)**(1-exp(-a*t))*log(Vmax/V0)
        
    def TStoLNprob(self, d, Vrate):
        #Schwarz's formula for THE PROBABILITY of an additional lymph node becoming involved
        #that year (!) given the current tumor volume and growth rate
        #(CISNET simulator adjusted the diameter by 75% after experiments)
        V = self.sphereV(d=0.75*d)
        return 0.0058+0.0052*V+0.0002*Vrate    
    
    def cancerProbs(self, age):
        #looks up probability of getting cancer given the age of the patient, returns this probability
        probs = self.pCancer
        return probs[str(age)]
    
    def deathProbs(self, age):
        #looks up probability of dying (all cause mortality) given the age of the patient, returns this probability
        probs = self.pDeath
        return probs[str(age)]
    
    def ERstatus(self, age):
        #simulates ER status given age, for person with cancer
        #based on CISNET TABLE 3. page 33
        if age < 45:
            p = 0.6
        if (age >= 45) & (age < 55):
            p = 0.65
        if (age >= 55) & (age < 65):
            p = 0.74
        if (age >= 65) & (age < 75):
            p = 0.77
        if age >= 75:
            p = 0.83
        return rbinom(1, p)
    

    def cancerStage(self, TS, LN):
         #translates tumor size and LN combo into cancer stage
         #0: no cancer
         #1: in situ
         #2: localised
         #3: regional
         #4: distant
         if TS == 0:
             stage = 0
         if (TS > 0) & (TS < 0.95) & (LN < 1):
             stage = 1
         if (TS >= 0.95) & (LN < 1):
             stage = 2
         if (LN >= 1) & (LN <= 4):
             stage = 3
         if LN >= 5:
             stage = 4
         return stage

    def initLN(self):
        #account for "hyper-agressive" growth cancers: 1% start with 4, 1% start with 5 lymph nodes from start
        rand = rmultinom(1, [0.01, 0.02, 0.97])
        if rand[0] == 1:
            initLNs = 4
        if rand[1] == 1:
            initLNs = 5
        if rand[2] == 1:
            initLNs = 0
        return initLNs

        
    def timeToDeath(self):
        #sample time till death (in Months) for distant stage cancer patients using empirical CDF (CISENT Fig3 p28)
        cdf= self.cdf
            
        def Xtrapolator(newY, X, Y):
            #function for extropolating a new X value given a discretely defined increasing function,
            #given by the ordered (X,Y) pairs. newY must be within Y range
            n = len(X)
            newX = -1 #error default
            for i in range(n-1):
                if newY < Y[0]:
                    newX = X[0]
                if newY > Y[n-1]:
                    newX = X[n-1]
                if Y[i] <= newY <= Y[i+1]:
                    newX = (newY-Y[i])*(X[i+1]-X[i])/(Y[i+1]-Y[i]) + X[i]
            return newX
                
        #inverse CDF method for sampling years to live after reaching distant stage cancer
        U = runif(0, 1)
        years = -1
        while years == -1:
            years = Xtrapolator(newY = U, X = cdf["years"], Y = cdf["probs"])
        return int(years*12)
    
    def stateEncoder(self, stage, TS, LN, age, ER):
        #first discretise all variables into numbered categories
        #dead or alive category
        if stage == 5:
            state = str(1) #dead from breast cancer
        elif stage == 6:
            state = str(2) #dead from all other causes
        else:
            state = str(0) #alive!
            #age  
            if age<50:
                state = state + str(0) #under 50
            elif 50 <= age < 60:
                state = state + str(1) #50-59
            elif 60 <= age < 70:
                state = state + str(2) #60-69
            else:
                state = state + str(3) #70 and above
            #cancer state
            if stage == 0:
                state = state + str(0) #patient doesn't have breast cancer
            else:
                state = state + str(1) #patient has breast cancer
                #tumour size, #lymph nodes, ER status combination
                #tumour size
                if (TS >= 0) & (TS <= 0.5):
                    tsCat = 0
                if (TS > 0.5) & (TS <= 0.95):
                    tsCat = 1
                if (TS > 0.95) & (TS <= 1.5):
                    tsCat = 2
                if (TS > 1.5) & (TS <= 2):
                    tsCat = 3
                if (TS > 2) & (TS <= 5):
                    tsCat = 4
                if (TS > 5) & (TS <= 9):
                    tsCat = 5
                #number of involved Lymph Nodes
                if LN < 1:
                    lnCat = 0
                if (LN >= 1) & (LN < 5):
                    lnCat = 1
                if LN >= 5:
                    lnCat = 2
                #Estrogen Receptor Status
                if ER < 1:
                    erCat = 0
                if ER == 1:
                    erCat = 1
                state = state + str(tsCat) + str(lnCat) + str(erCat)
        return state
    
    def stateNumber(self, statecode):
        #takes in statecode string, converts to the state reference number (integer from 0 to 149)
        stateDict = self.stateDict
        return stateDict[statecode]
    
    def simHistory(self):
        ##simulate patient's tumor size history
        months = self.months
        #generate binary time series indicating 1: cancer positive, 0: cancer negative, 2: death of other causes
        tum = np.zeros(len(months))
        for i in range(len(months)-1):
            if tum[i] == 1:
                tum[i+1] = 1
            #every year, test for cancer initiation and death    
            elif i%12 == 0:
                age = int(months[i]/12)
                pC = self.cancerProbs(age = age)
                if rbinom(1, pC) == 1:
                    month = randint(1, 12)
                    tum[i+month] = 1
                
        #simulate ER status, tumor growth and lymph nodes involved from point of tumor initiation (if it happens)        
        ER = -1 #default value (-1 means no cancer, 0 means cancer but ER negative, 1 means cancer but ER positive)
        tonset = -1 #default value for onset time, -1 means never 
        tumD = np.zeros(len(months))
        growthrate = np.zeros(len(months))
        LN = np.zeros(len(months))
        #find time of initiation (!in months after StartAge!) if it happens
        if len(np.where(tum == 1)[0]) != 0:
            init = np.where(tum == 1)[0][0]
            tonset = self.startAge*12 + init
            #sample ER status based on onset age (in years)
            ER = self.ERstatus(age = int(tonset/12))
            tumD[init] = self.d0
            self.genAlpha()
            growthrate[init] = self.alpha
            tumD[init+1:len(months)] = [self.tumSize(t=t, a=self.alpha) for t in range(1, len(months)-init)]
            growthrate[init+1:len(months)] = [self.tumSizeRate(t=t, a=self.alpha) for t in range(1, len(months)-init)]
            #simulate LNs invloved
            #account for non malignant tumours called LMP = stop growing at 1cm then regress (disappear after 2 years)
            #sample LMP status -> 42% of tumours are LMP
            LMP = rbinom(1, 0.42)
            if LMP == 0:
                LN[init] = self.initLN() #hyper aggressive tumours are part of non-LMP only
            else:
                LN[init] = 0
            for t in range(init+1, len(months)):
                if t%12 == 0:
                    pLN = self.TStoLNprob(d=tumD[t], Vrate = growthrate[t])
                    addLN = rbinom(1, pLN)
                    LN[t] = LN[t-1] + addLN
                else:
                    LN[t] = LN[t-1]
            
            if LMP == 1:
                tumD[init+1:len(months)] = [min(d, 1) for d in tumD[init+1:len(months)]]
                
                if len(np.where(np.array(tumD) == 1)[0]) != 0:
                    
                    tmaxLMP = np.where(np.array(tumD) == 1)[0][0]
                    LN = [0 for num in LN]
                    if tmaxLMP+24 < len(months):
                        tumD[tmaxLMP+24:len(months)] = [0 for d in tumD[tmaxLMP+24:len(months)]]
                  
        onsetAge = int(tonset/12)        
                
        self.ER = ER
        self.TShistory = tumD
        self.LNhistory = LN
        #produce cancer stage time series
        stage = np.array([self.cancerStage(TS = self.TShistory[t], LN = self.LNhistory[t]) for t in range(len(months))])
        #add death all cause state
        ACDtime = self.endAge*12 #default if death by all cause doesnt occur
        for i in range(len(months)-1):
            if stage[i] == 6:
                stage[i+1] = 6
            elif i%12 == 0:
                age = int(months[i]/12)
                pD = self.deathProbs(age = age)
                deathYN = rbinom(1, pD) #did patient die the next year YES/NO?
                if deathYN == 1:
                    month = randint(1,12)
                    stage[i+month] = 6 #stage = 6: death from all causes
                    ACDtime = months[i+month] #All Cause Death time in months
        
        #add death from cancer (state 5)
        CDtime = self.endAge*12 #default if death by cancer doesnt occur
        tleft = self.timeToDeath() #only used if patient reaches stage IV but must draw outside loop
        switch = 0 #elif statement only executed once, at the 1st timestep the patient reaches stage IV
        for i in range(len(months)-1):
            if stage[i] == 5:
                stage[i+1] = 5
            elif (stage[i] == 4) & (switch == 0):
                if i+tleft < len(months):
                    if stage[i+tleft] != 6:
                        stage[i+tleft] = 5
                        CDtime = months[i+tleft] #Cancer Death time in months
                        switch = 1
                
        self.stagehistory = stage
        self.deathAge = int(min(ACDtime, CDtime)/12)
        
        if onsetAge >= self.deathAge:
            onsetAge = -1
        self.onsetAge = onsetAge
        
        self.states = [self.stateEncoder(stage[t], tumD[t], LN[t], int(months[t]/12), ER) for t in range(len(months))]
        self.statenums = [self.stateNumber(state) for state in self.states]