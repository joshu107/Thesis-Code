# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 18:04:49 2018

@author: Joshua
"""

from urllib.request import urlopen
import re
import pandas as pd
import pickle
import numpy as np


home = "C:/Users/jbhud/Documents/Thesis/Thesis Code/"

#I. CANCER PROBS

probs = []
ages = []
for age in range(35, 90):
    
    url = "https://www.cancer.gov/bcrisktool/RiskAssessment.aspx?genetics=2"
    url += "&current_age=" + str(age)
    url += "&age_at_menarche=99&age_at_first_live_birth=99&ever_had_biopsy=99&previous_biopsies=99&biopsy_with_hyperplasia=99&related_with_breast_cancer=99&race=6"
    response = urlopen(url)
    webContent = response.read().decode("utf8")
    ages = ages + [str(age)]
    probs = probs + re.findall("<span id=\"ctl00_cphMain_lbl5YrAbsoluteRisk\">([\s\S]*?)%</span>", webContent)
    
probs = [float(p) for p in probs]


df = pd.DataFrame(data = probs, index = ages, columns = ["prob"])

with open(home+"1. Simulation/cancerprobs.pkl", "wb") as f:
    pickle.dump(df, f)
    
#II. DEATH PROBS (ALL CAUSE)
df2 = pd.read_excel(home+"1. Simulation/deathprobs.xlsx",
                    header = 0, index_col = 0)
df2.index = [str(i) for i in df.index]
with open(home+"1. Simulation/deathprobs.pkl", "wb")as f:
    pickle.dump(df2, f)


#III. CANCER DEATH PROBS (DISTRIBUTION OF TIME TILL DEATH FROM STAGE 4 CANCER)

df3 = pd.read_csv(home+"1. Simulation/cancersurvivalprobs.csv",
                  names = ["years", "probs"])

#float range function
def myrange(start, stop, step):
    i = start
    res = []
    while i < stop:
         res += [i]
         i += step
    return res
         
df3["years"] = myrange(0, len(df3["years"])/2, 0.5)
diff = np.zeros(len(df3["years"])-1)
for i in range(len(df3["years"])-1):
     diff[i] = df3["probs"].iloc[i+1] - df3["probs"].iloc[i]

np.where(diff <= 0)[0][0] #38 is a point where CDF decreasing according to my coordinate getting with https://apps.automeris.io/wpd/
# extrapolate (mean of neighbouring points) and replace this value
df3["probs"].iloc[38] = (df3["probs"].iloc[37] + df3["probs"].iloc[39])/2

#save as pickle
with open(home+"1. Simulation/cancerdeathtimeprobs.pkl", "wb")as f:
    pickle.dump(df3, f)