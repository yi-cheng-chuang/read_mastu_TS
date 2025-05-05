# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:56:49 2025

@author: ychuang
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import mtanh_fitting as mf

# Replace 'your_file.pkl' with the path to your pickle file
file_path = 'TS_pedestal_49404.pkl'

# Open and load the pickle file
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Display the loaded data
print(type(data))
print(data[49404].keys())
print(data[49404]['laser_time'])
print(data[49404]['Te_arrays']['Pulse_0.799_0.830'].keys())
print(data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers'].keys())
time_list = list(data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers'].keys())

print(data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers']['0.80614'].keys())


dlcfs = data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers']['0.80614']['dlcfs']
print(dlcfs)



def nan_filter(pos, ne, te):
    
    ne_output = []
    te_output = []
    pos_output = []
    
    
    for kk, item in enumerate(ne):
        if np.isnan(item) == False and np.isnan(te[kk]) == False:
            ne_output.append(item)
            te_output.append(te[kk])
            pos_output.append(pos[kk])
    
    data_dic = {'dlcfs': pos_output, 'ne': ne_output, 'te': te_output}
    
    return data_dic


def LFS_cutoff(dlcfs, ne, te):
    
    dlcfs_list = []
    ne_list = []
    te_list = []
    
    for kk, it in enumerate(dlcfs):
        
        if it >= 1.25 and np.isnan(ne[kk]) == False and np.isnan(te[kk]) == False:
            dlcfs_list.append(it)
            ne_list.append(ne[kk])
            te_list.append(te[kk])
    
    dlcfs_pro = np.array(dlcfs_list)
    ne_pro = np.array(ne_list)
    te_pro = np.array(te_list)
    
    datac_dic = {'dlcfs': dlcfs_pro, 'ne': ne_pro, 'te': te_pro}
    
    return datac_dic


def LFS_bin_cutoff(dlcfs, ne, te, ne_error, te_error):
    
    dlcfs_list = []
    ne_list = []
    te_list = []
    ne_error_list = []
    te_error_list = []
    
    for kk, it in enumerate(dlcfs):
        
        if it >= -0.2 and np.isnan(ne[kk]) == False and np.isnan(te[kk]) == False:
            dlcfs_list.append(it)
            ne_list.append(ne[kk])
            te_list.append(te[kk])
            ne_error_list.append(ne_error[kk])
            te_error_list.append(te_error[kk])
    
    dlcfs_pro = np.array(dlcfs_list)
    ne_pro = np.array(ne_list)
    te_pro = np.array(te_list)
    ne_error_pro = np.array(ne_error_list)
    te_error_pro = np.array(te_error_list)
    
    datac_dic = {'dlcfs': dlcfs_pro, 'ne': ne_pro, 'te': te_pro, 
                 'ne_error': ne_error_pro, 'te_error': te_error_pro}
    
    return datac_dic





alldata_dic = {}

# for time in time_list:
    
#     target_dic = data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers'][time]
#     # radius = target_dic['radius']
#     dlcfs = target_dic['dlcfs']
#     te = target_dic['te']
#     ne = target_dic['ne']
#     # print(type(ne))
#     datac_dic = LFS_cutoff(dlcfs = dlcfs, ne = ne, te = te)
    
#     alldata_dic[time] = datac_dic
    
    

for time in time_list:
    
    target_dic = data[49404]['Te_arrays']['Pulse_0.799_0.830']['lasers'][time]
    # radius = target_dic['radius']
    radius = target_dic['radius']
    te = target_dic['te']
    ne = target_dic['ne']
    # print(type(ne))
    datac_dic = LFS_cutoff(dlcfs = radius, ne = ne, te = te)
    
    alldata_dic[time] = datac_dic
            

"prepare for flatten"

ln = len(time_list)

an_list = []

for ti in time_list:
    
    an_list.append(len(alldata_dic[ti]['dlcfs']))
    

an = max(an_list)

print(an)

dlcfs_flat = []
ne_flat = []
te_flat = []

for tm in time_list:
    
    for kb, d in enumerate(alldata_dic[tm]['dlcfs']):
        
        dlcfs_flat.append(d)
        ne_flat.append(alldata_dic[tm]['ne'][kb])
        te_flat.append(alldata_dic[tm]['te'][kb])
    
    


"try to turn into array"


# print(dlcfs_flat)

plt.figure()

plt.scatter(dlcfs_flat, ne_flat, label= 'ne_flatter')

# Add labels and title
plt.xlabel("dlcfs (m)", fontsize=12)
plt.title("ne vs. dlcfs", fontsize=14)

# Add legend
plt.legend()

# Show the plot
plt.show()


# print(dlcfs_flat)



def tanh(r,r0,h,d,b,m):
    return b+(h/2)*(np.tanh((r0-r)/d)+1) + m*(r0-r-d)*np.heaviside(r0-r-d, 1)


"""

[1.39320728e+00 6.09502696e-02 4.51153933e-04 1.32121883e-02
 2.99272370e+00]

[pedestal center position, pedestal full width, Pedestal top, Pedestal bottom, 
 inboard slope, inboard quadratic term, inboard cubic term, outboard linear term,
 outbard quadratic term]

[1.37, 0.25, 0.3, 0.01, 0.02, 0.02, 0.02, 0.01, 0.01]

"""


def Osbourne_Tanh(x,C,reg=None,plot = False):
    """
    adapted from Osborne via Hughes idl script
    tanh function with cubic or quartic inner and linear
    to quadratic outer extensions and 

    INPUTS: 
    c = vector of coefficients defined as such:
    c[0] = pedestal center position
    c[1] = pedestal full width
    c[2] = Pedestal top
    c[3] = Pedestal bottom
    c[4] = inboard slope
    c[5] = inboard quadratic term
    c[6] = inboard cubic term
    c[7] = outboard linear term
    c[8] = outbard quadratic term
    x = x-axis
    """

    if reg is not None:
        for each_coeff in reg:
            C[each_coeff] = 0

    z = 2. * ( C[0] - x ) / C[1]
    P1 = 1. + C[4] * z + C[5] * z**2 + C[6] * z**3
    P2 = 1. + C[7] * z + C[8] * z**2
    E1 = np.exp(z)
    E2 = np.exp(-1.*z)
    F = 0.5 * ( C[2] + C[3] + ( C[2] - C[3] ) * ( P1 * E1 - P2 * E2 ) / ( E1 + E2 ) )

    if plot:
        f, a = plt.subplots(1,1)
        a.plot(x,F)
        plt.show()

    return F




n_tot = 200
dlcfs_pro = np.linspace(min(dlcfs_flat), max(dlcfs_flat), num= n_tot, dtype=float)

Ne = [x *pow(10, -19) for x in ne_flat]
Te = [x*pow(10, -3) for x in te_flat]

x_arr = np.asarray(dlcfs_flat)
te_arr = np.asarray(Te)


p0 = [1.4, 0.6, 0.01, 0.01, 3/14]
p1 = [1.4, 0.6, 0.01, 0.01, 3/14]

popt_ne, pcov_ne = curve_fit(tanh, dlcfs_flat, Ne, p0)      
popt_te, pcov_te = curve_fit(tanh, dlcfs_flat, Te, p1)


print(popt_ne)
print(popt_te)


ne_fit = tanh(dlcfs_pro, popt_ne[0], popt_ne[1], popt_ne[2], 
                 popt_ne[3], popt_ne[4])*pow(10, 19)
te_fit = tanh(dlcfs_pro, popt_te[0], popt_te[1], popt_te[2], 
                 popt_te[3], popt_te[4])*pow(10, 3)

ts_start = 0.799  # Start TS time [s]
ts_end = 0.83

# Create a unique key for the pulse using the specified start and end times.
pulse_key = f'Pulse_{ts_start:.3f}_{ts_end:.3f}'



plt.figure()

plt.scatter(dlcfs_flat, ne_flat, label= f'ne_{pulse_key}')
plt.plot(dlcfs_pro, ne_fit, label= 'ne_tanh_fit', color = 'red')

# Add labels and title
plt.xlabel("radius (m)", fontsize=12)
plt.title("ne vs. radius", fontsize=14)

# Add legend
plt.legend()

# Show the plot
plt.show()



plt.figure()

plt.scatter(dlcfs_flat, te_flat, label= f'te_{pulse_key}')
plt.plot(dlcfs_pro, te_fit, label= 'te_tanh_fit', color = 'red')

# Add labels and title
plt.xlabel("radius (m)", fontsize=12)
plt.title("te vs. radius", fontsize=14)

# Add legend
plt.legend()

# Show the plot
plt.show()


writefile = True

xrange = len(x_arr)
te_range = len(te_arr)

print(xrange)
print(te_range)


if writefile:
    w_datalist = []
    filename = '49404_82.dat'
    
    for j in range(xrange):
        w_list =[]
        w_list.append("{: .3f}".format(x_arr[j]))
        w_list.append("{: .3f}".format(te_arr[j]))
        w_writelist = ' '.join(str(y)+ "\t" for y in w_list)
        w_datalist.append(w_writelist)
   
    with open(filename,'w') as f:
        for l,w_line in enumerate(w_datalist):   
            f.writelines(w_line + "\n")




['Te_arrays', 'dlcsf_te', 'te_data', 'dlcs_bins', 'te_bins', 'te_bins_error', 'ne_data', 'ne_bins', 'ne_bins_error', 'laser_time']


dlcfs_bins = data[49404]['dlcs_bins']
ne_bins = data[49404]['ne_bins']
ne_error = data[49404]['ne_bins_error']
te_bins = data[49404]['te_bins']
te_error = data[49404]['te_bins_error']


databin_dic = LFS_bin_cutoff(dlcfs = dlcfs_bins, ne = ne_bins, te = te_bins, 
                             ne_error = ne_error, te_error = te_error)

bins_plot = False
plot = False



if bins_plot:
    
    plt.figure()

    plt.errorbar(dlcfs_bins, ne_bins, yerr = ne_error, fmt = 'o', label= f'{time}s')

    # Add labels and title
    plt.xlabel("dlcfs (m)", fontsize=12)
    plt.title("ne vs. dlcfs", fontsize=14)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


    plt.figure()

    plt.errorbar(dlcfs_bins, te_bins, yerr = te_error, fmt = 'o', label= f'{time}s')

    # Add labels and title
    plt.xlabel("dlcfs (m)", fontsize=12)
    plt.title("te vs. dlcfs", fontsize=14)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
    


if plot:
    
    plt.figure()

    for time in time_list:
        
        dlcfs_plot = alldata_dic[time]['dlcfs']
        ne_plot = alldata_dic[time]['ne']
        
        
        plt.scatter(dlcfs_plot, ne_plot, label= f'{time}s', linewidth=2)


    # Add labels and title
    plt.xlabel("dlcfs (m)", fontsize=12)
    plt.title("ne vs. dlcfs", fontsize=14)

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()



    plt.figure()

    for time in time_list:
        
        dlcfs_plot = alldata_dic[time]['dlcfs']
        ne_plot = alldata_dic[time]['te']
        
        
        plt.scatter(dlcfs_plot, ne_plot, label= f'{time}s', linewidth=2)


    # Add labels and title
    plt.xlabel("dlcfs (m)", fontsize=12)
    plt.title("te vs. dlcfs", fontsize=14)


    # Add legend
    plt.legend()

    # Show the plot
    plt.show()   







    

