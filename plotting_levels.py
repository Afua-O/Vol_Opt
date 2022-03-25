# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:57:24 2022

@author: aow001
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from platypus import Problem, EpsNSGAII, Real, ProcessPoolEvaluator
import csv
import logging
from itertools import chain
logging.basicConfig(level=logging.INFO)
plt.rcParams['figure.figsize'] = [12, 8]
# sys.path.append('..')


sys.path.append(os.path.abspath('..'))
from Volta_model_4 import VoltaModel
import rbf_functions
# output_dir = f"./output/{rbf}/"

entry = rbf_functions.squared_exponential_rbf
#name = entry.__name__


# load variables from refset
varlist = []
variables = []
output_dir = os.path.abspath('../Vol_Opt/Saved solutions/clam_his')  #change for each plot
for filename in os.listdir(output_dir):
    if filename == f'10_variables.csv':
        varlist.append(filename[:-4])
        df_vars = pd.read_csv(f"{output_dir}\{filename}")
        
variables = df_vars.values
        
"""
for filename in os.listdir('../data1999'):
    if filename.startswith('w'):
        globals()[f"{filename[:-4]}"] = np.loadtxt(f'../data1999/{filename}')
        
print(f"Loaded: {', '.join(varlist)}")
"""


n_inputs = 2  # (time, storage of Akosombo)
n_outputs = 2 # Irrigation, Downstream:- (hydropower, environmental, floodcontrol)
n_rbfs = n_inputs+2
rbf = rbf_functions.RBF(n_rbfs, n_inputs, n_outputs, rbf_function=entry)

# Initialize model
nobjs = 5
n_years = 1
lowervolta_river = VoltaModel(241.0, 505.0, n_years, rbf)
lowervolta_river.set_log(True)
# lowervolta_river.setRBF(numberOfRBF, numberOfInput, numberOfOutput, RBFType)
output=[]
for dvars in variables:
    output.append(lowervolta_river.evaluate(dvars))
    
level_Ak, rirri, renv = lowervolta_river.get_log()


pd.DataFrame(data=output, columns=[['hydropowerAk','irrigation','environment','floodcontrol','hydropowerKp']])


from Volta_model_4 import create_path

if not os.path.exists(f'figs/releases/clam_his'):  #change for each plot
    os.makedirs(f'figs/releases/clam_his')   

alpha = 0.1
lw = 0.5                                

for year in level_Ak:
   plot = plt.plot(year, "blue", linewidth=lw, alpha=alpha)
   plot = plt.xlabel('days')
   plot = plt.ylabel('Ak_level_ft')
   plot = plt.ylim((238, 278))
plt.savefig(f'figs/releases/clam_his/clamhis_levelak.jpg')  #change for each plot
plt.show()                