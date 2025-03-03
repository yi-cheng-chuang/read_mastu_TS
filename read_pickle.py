# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:56:49 2025

@author: ychuang
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'TS_pedestal_49404.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Display the loaded data
print(type(data))
print(data[49404].keys())
print(data[49404]['Te_arrays']['Pulse_0.820_0.840'].keys())
print(data[49404]['Te_arrays']['Pulse_0.820_0.840']['lasers']['0.82222'].keys())

target_dic = data[49404]['Te_arrays']['Pulse_0.820_0.840']['lasers']['0.82222']

print(target_dic['time'])

radius = target_dic['radius']
dlcfs = target_dic['dlcfs']
te = target_dic['te']
ne = target_dic['ne']

# Create the plot
plt.figure()
plt.scatter(radius, ne, label='ne', color='b', linewidth=2)

# Add labels and title
plt.xlabel("radius (m)", fontsize=12)
plt.title("ne vs. radius", fontsize=14)


# Create the plot
plt.figure()
plt.scatter(dlcfs, ne, label='ne', color='b', linewidth=2)

# Add labels and title
plt.xlabel("dlcfs (m)", fontsize=12)
plt.title("ne vs. dlcfs", fontsize=14)


# Add legend
plt.legend()

# Show the plot
plt.show()
