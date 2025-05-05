# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:02:52 2025

@author: ychuang
"""


import os
import pickle
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pyuda


# =============================================================================
# Database and CSV Utilities
# =============================================================================
def is_shot_in_database(shot, pickle_file_path):
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            database = pickle.load(file)
            if shot in database:
                return True
    return False

def read_good_shots_from_csv(file_path):
    good_shots = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            good_shots.append(int(row[0]))
    return good_shots

def update_database(shot, database, pickle_file_path):
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(database, file)

# =============================================================================
# TS Data Processing Functions
# =============================================================================
def get_data_trace(trace_name, shot, start_t, end_t):
    """
    Retrieve a data trace for a given shot between start_t and end_t.
    
    Returns:
        tuple: (trace_time, trace_data, error_code)
    """
    error_code = 0
    try:
        trace_file = client.get(trace_name, shot)
        print(f"Reading shot {shot}, channel {trace_name}")
        trace_data = trace_file.data
        trace_time = trace_file.time.data
        # Select indices closest to start_t and end_t
        start_pos = np.argmin(np.abs(trace_time - start_t))
        end_pos = np.argmin(np.abs(trace_time - end_t))
        trace_data = trace_data[start_pos:end_pos]
        trace_time = trace_time[start_pos:end_pos]
    except pyuda.UDAException as err:
        print(f"No data for shot {shot}, channel {trace_name}. Error: {err}")
        error_code = 1
        trace_time = np.zeros(1)
        trace_data = np.zeros(1)
    return trace_time, trace_data, error_code

def func_get_ts_times_rads(shot, ts_times, t1, t2):
    """
    Retrieve TS times and corresponding radius data at specific TS times,
    using a uniform grid while avoiding duplicate indices.
    
    Returns:
        tuple: (t_r, d_r, error_code, t_args, ts_actual_times, radius_data)
    """
    num_times = ts_times.shape[0]
    t_r, d_r, error_code = get_data_trace('/AYC/R', shot, t1, t2)
    t_args = np.zeros(num_times, dtype=int)
    ts_actual_times = np.zeros(num_times)
    num_rads = d_r.shape[1]
    radius_data = np.zeros((num_times, num_rads))
    used_indices = set()
    if error_code == 0:
        for z in range(num_times):
            differences = np.abs(t_r - ts_times[z])
            for idx in used_indices:
                differences[idx] = np.inf
            arg_pos = np.argmin(differences)
            used_indices.add(arg_pos)
            t_args[z] = arg_pos
            ts_actual_times[z] = t_r[arg_pos]
            radius_data[z, :] = d_r[arg_pos, :]
    return t_r, d_r, error_code, t_args, ts_actual_times, radius_data

def func_get_ts_times_rads_range(shot, ts_start, ts_end, t1, t2):
    """
    Retrieve TS times, radius, Tₑ and nₑ data over a given time range.
    
    Returns:
        tuple: (ts_times, radius_data, te_data, ne_data)
    """
    t_r, d_r, error_code = get_data_trace('/AYC/R', shot, t1, t2)
    arg_1 = np.argmin(np.abs(t_r - ts_start))
    arg_2 = np.argmin(np.abs(t_r - ts_end))
    ts_times = t_r[arg_1:arg_2]
    radius_data = d_r[arg_1:arg_2, :]
    t_te, d_te, error_code_te = get_data_trace('/AYC/T_E', shot, t1, t2)
    if error_code_te == 0:
        te_data = d_te[arg_1:arg_2, :]
    else:
        te_data = np.zeros_like(radius_data)
    t_ne, d_ne, error_code_ne = get_data_trace('/AYC/N_E', shot, t1, t2)
    if error_code_ne == 0:
        ne_data = d_ne[arg_1:arg_2, :]
    else:
        ne_data = np.zeros_like(radius_data)
    return ts_times, radius_data, te_data, ne_data


def func_get_z(shot, ts_start, ts_end, t1, t2):
    
    t_r, d_r, error_code = get_data_trace('/AYC/R', shot, t1, t2)
    arg_1 = np.argmin(np.abs(t_r - ts_start))
    arg_2 = np.argmin(np.abs(t_r - ts_end))
    ts_times = t_r[arg_1:arg_2]
    radius_data = d_r[arg_1:arg_2, :]
    t_z, d_z, error_code = get_data_trace('/AYC/Z', shot, t1, t2)
    arg_3 = np.argmin(np.abs(t_z - ts_start))
    arg_4 = np.argmin(np.abs(t_z - ts_end))
    tz_times = t_z[arg_3:arg_4]
    z_data = d_z[arg_3:arg_4, :]
    
    
    
    # Retrieve EFIT data for plasma radius
    trace_name_efit = '/EPM/OUTPUT/SEPARATRIXGEOMETRY/RMIDPLANEOUT'
    t_efr, efr_dat, error_code_efr = get_data_trace(trace_name_efit, shot, t1, t2)
    if error_code_efr == 0:
        valid_idx = func_nans_removed(efr_dat)
        t_efr = t_efr[valid_idx]
        efr_dat = efr_dat[valid_idx]
    else:
        print("Error retrieving EFIT data.")
    
    # Interpolate plasma radius at TS times (from the full range) using EFIT data.
    ts_plasma_r = np.interp(ts_times, t_efr, efr_dat)
    
    

    return ts_times, radius_data, tz_times, z_data

def func_nans_removed(test_array):
    return np.logical_not(np.isnan(test_array))

def from_epm(shot, ts_start, ts_end, t1, t2, ts_times):
    
    
    # Retrieve EFIT data for plasma radius
    trace_name_efit = '/EPM/OUTPUT/SEPARATRIXGEOMETRY/RMIDPLANEOUT'
    t_efr, efr_dat, error_code_efr = get_data_trace(trace_name_efit, shot, t1, t2)
    if error_code_efr == 0:
        valid_idx = func_nans_removed(efr_dat)
        t_efr = t_efr[valid_idx]
        efr_dat = efr_dat[valid_idx]
    else:
        print("Error retrieving EFIT data.")
    
    # Interpolate plasma radius at TS times (from the full range) using EFIT data.
    ts_plasma_r = np.interp(ts_times, t_efr, efr_dat)
    
    
    
    # Retrieve EFIT data for plasma radius
    trace_name_efit = '/EPM/OUTPUT/SEPARATRIXGEOMETRY/ZBOUNDARY'
    t_ezb, ezb_dat, error_code_ezb = get_data_trace(trace_name_efit, shot, t1, t2)
    if error_code_ezb == 0:
        valid_idx = func_nans_removed(efr_dat)
        t_efr = t_efr[valid_idx]
        efr_dat = efr_dat[valid_idx]
    else:
        print("Error retrieving EFIT data.")
    
    # Interpolate plasma radius at TS times (from the full range) using EFIT data.
    ts_plasma_r = np.interp(ts_times, t_efr, efr_dat)
    
    
    
    # Retrieve EFIT data for plasma radius
    trace_name_efit = '/EPM/OUTPUT/SEPARATRIXGEOMETRY/ZGEOM'
    t_efr, efr_dat, error_code_efr = get_data_trace(trace_name_efit, shot, t1, t2)
    if error_code_efr == 0:
        valid_idx = func_nans_removed(efr_dat)
        t_efr = t_efr[valid_idx]
        efr_dat = efr_dat[valid_idx]
    else:
        print("Error retrieving EFIT data.")
    
    # Interpolate plasma radius at TS times (from the full range) using EFIT data.
    ts_plasma_r = np.interp(ts_times, t_efr, efr_dat)
    
    



def main():
    """
    Main function to process TS data for a given shot and save data for a user-specified pulse.
    The pulse is defined by a time range (ts_start to ts_end). Within the pulse,
    the individual laser (TS) signals are stored in a sub-dictionary for later analysis.
    For each laser measurement, DLCFS is computed and stored.
    """
    start_time = datetime.now()
    plt.close('all')
    
    global client
    client = pyuda.Client()
    
    # ----------------------------
    # User/Shot Configuration
    # ----------------------------
    shot = 49404         # Shot number to process
    t1 = 0.1             # Global start time [s]
    t2 = 1.5             # Global end time [s]
    

    ts_start = 0.799  # Start TS time [s]
    ts_end = 0.83     # End TS time [s]
    
    ts_times, radius_data, tz_times, z_data = func_get_z(shot, ts_start, ts_end, t1, t2)
    
    # ----------------------------
    # Update Database with Pulse Data
    # ----------------------------
    pickle_file_path = f'TS_pedestal_getZ_{shot}.pkl'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            db = pickle.load(file)
    else:
        db = {}
    
    # Create a unique key for the pulse using the specified start and end times.
    pulse_key = f'Pulse_{ts_start:.3f}_{ts_end:.3f}'
    
    
    
    
    
    shot_data = {'ts_times': ts_times,'radius_data': radius_data, 
                 'tz_times': tz_times, 'z_data': z_data}
    
    sep_data = 
    
    

    db[shot] = shot_data
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(db, file)
    print(f"Data for shot {shot}, pulse {pulse_key} has been saved to the database.")

    # if os.path.exists(pickle_file_path):
    #     with open(pickle_file_path, 'rb') as file:
    #         shots_data = pickle.load(file)
    #     print("Database loaded successfully.")
    # else:
    #     print("Pickle file not found.")

    end_time = datetime.now()
    print("Processing completed. Duration:", end_time - start_time)



if __name__ == '__main__':
    results_dict = main()


    
    
    
    