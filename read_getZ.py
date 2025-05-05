# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:40:18 2025

@author: ychuang
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'TS_pedestal_getZ_49404.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Display the loaded data
print(type(data))
print(data[49404].keys())
print(data[49404]['z_data'])
# print('radius data')
# print(data[49404]['radius_data'])
