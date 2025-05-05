#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Automatic Grouping of Thomson Scattering Laser Pulses

This script retrieves TS laser data over a user‐specified time window and automatically
groups measurements into pulses. A pulse is defined as a group of laser measurements that
occur closely in time (with adjacent measurements separated by less than a threshold, e.g. 5 ms)
and separated from the next group by a gap larger than the threshold.

For each pulse, the script:
  • Retrieves the TS time and radius data from the '/AYC/R' channel.
  • Retrieves the corresponding Tₑ and nₑ data from '/AYC/T_E' and '/AYC/N_E' (by matching times).
  • Retrieves EFIT data for the plasma radius and computes the DLCFS (difference between the
    measured radius and the interpolated plasma radius) for each laser measurement.
  • Stores all the per‑laser data (time, radius, Tₑ, nₑ, DLCFS) in a sub‐dictionary keyed by the
    laser time, and then stores the complete pulse (with its start and end time) in a “pulses” dictionary.
  
Finally, the pulses are saved in a database (pickle file) so that later you can choose to analyze
only the pulses (or even a specific laser channel across pulses) that you are interested in.

Created on Mon Feb  3 16:51:37 2025
@author: qf9176
"""

import os
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pyuda

# =============================================================================
# Database Utilities
# =============================================================================
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
      (trace_time, trace_data, error_code)
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

def func_nans_removed(test_array):
    return np.logical_not(np.isnan(test_array))

def func_curr_density(te_data, ne_data):
    e = 1.6e-19      # Elementary charge [C]
    mass_d = 2 * 1.6726e-27  # Deuterium mass [kg]
    pin_r = 0.75e-3  # Pin radius [m]
    pin_h = 2e-3     # Pin height [m]
    pin_area = np.pi * pin_r**2 + 2 * np.pi * pin_r * pin_h
    j0 = e**(1.5) * mass_d**(-0.5) * ne_data * np.sqrt(te_data)
    i0 = j0 * pin_area
    return j0, i0

def func_bin(x_data, y_data, x_bins):
    y_data[np.isinf(y_data)] = np.nan
    binned_data = np.digitize(x_data, x_bins)
    num_bins = x_bins.shape[0]
    bins_info = np.zeros((num_bins, 4))
    for a in range(num_bins):
        bin_indices = (binned_data == a)
        bm = np.nanmean(y_data[bin_indices])
        bs = np.nanstd(y_data[bin_indices])
        bc = np.sum(bin_indices)
        bins_info[a, :] = np.array([x_bins[a], bm, bs, bc])
    return bins_info

# =============================================================================
# Automatic Pulse Grouping Function
# =============================================================================
def group_pulses_auto(t_array, threshold):
    """
    Group measurements into pulses based on a time difference threshold.
    
    Parameters:
      t_array (np.array): Sorted 1D array of TS times.
      threshold (float): Time gap threshold (in seconds) to define separation between pulses.
      
    Returns:
      list of lists: Each sub-list contains the indices (in t_array) that belong to one pulse.
    """
    groups = []
    current_group = [0]
    for i in range(len(t_array) - 1):
        if t_array[i+1] - t_array[i] < threshold:
            current_group.append(i+1)
        else:
            groups.append(current_group)
            current_group = [i+1]
    if current_group:
        groups.append(current_group)
    return groups

# =============================================================================
# Main Processing and Plotting
# =============================================================================
def main():
    """
    Main function to process TS data for a shot over a search window, automatically
    grouping laser measurements into pulses and saving all pulses (with per-laser data)
    in a database.
    """
    start_time = datetime.now()
    plt.close('all')
    
    global client
    client = pyuda.Client()
    
    # ----------------------------
    # User/Shot Configuration
    # ----------------------------
    shot = 49404         # Shot number to process
    t1 = 0.1             # Global start time of shot [s]
    t2 = 1.5             # Global end time of shot [s]
    
    # Define the overall time window to search for laser pulses.
    # (Adjust these to restrict the search to a specific period.)
    search_start = 0.785  # e.g., start of TS pulse activity [s]
    search_end   = 0.85  # e.g., end of TS pulse activity [s]
    
    # Define the threshold to separate pulses (e.g., 5 ms)
    pulse_threshold = 0.010  # seconds
    
    # ----------------------------
    # Data Retrieval for Full Search Window
    # ----------------------------
    # Retrieve full TS time and radius data from '/AYC/R' over the search window.
    full_t_r, full_d_r, err = get_data_trace('/AYC/R', shot, search_start, search_end)
    if err:
        print("Error retrieving TS radius data. Exiting.")
        return

    # Ensure full_t_r is sorted (should be, but just in case)
    sort_idx_full = np.argsort(full_t_r)
    full_t_r = full_t_r[sort_idx_full]
    full_d_r = full_d_r[sort_idx_full, :]
    
    # Automatically group measurements into pulses.
    pulse_groups = group_pulses_auto(full_t_r, pulse_threshold)
    print(f"Found {len(pulse_groups)} pulses in the search window.")
    
    # Retrieve Tₑ and nₑ data over the same search window.
    t_te, d_te, err_te = get_data_trace('/AYC/T_E', shot, search_start, search_end)
    t_ne, d_ne, err_ne = get_data_trace('/AYC/N_E', shot, search_start, search_end)
    
    # Retrieve EFIT plasma radius data (for DLCFS computation).
    t_efr, efr_dat, err_efr = get_data_trace('/EPM/OUTPUT/SEPARATRIXGEOMETRY/RMIDPLANEOUT', shot, search_start, search_end)
    if err_efr:
        print("Warning: Error retrieving EFIT data; DLCFS will be set to zero.")
        t_efr = np.array([search_start, search_end])
        efr_dat = np.zeros_like(t_efr)
    
    # ----------------------------
    # Process Each Pulse Group
    # ----------------------------
    pulses = {}  # Dictionary to store all pulses.
    for i, group in enumerate(pulse_groups, start=1):
        group_times = full_t_r[group]           # TS times for this pulse.
        group_radius = full_d_r[group, :]         # Corresponding radius data.
        
        # For each measurement in the group, find the closest Tₑ and nₑ data.
        group_te = []
        group_ne = []
        group_dlcfs = []
        for t_val, rad in zip(group_times, group_radius):
            # Find closest Tₑ index:
            idx_te = np.argmin(np.abs(t_te - t_val)) if err_te==0 else 0
            group_te.append(d_te[idx_te])
            # Find closest nₑ index:
            idx_ne = np.argmin(np.abs(t_ne - t_val)) if err_ne==0 else 0
            group_ne.append(d_ne[idx_ne])
            # Compute DLCFS: difference between measured radius and plasma radius.
            plasma_radius = np.interp(t_val, t_efr, efr_dat)
            group_dlcfs.append(rad - plasma_radius)
        group_te = np.array(group_te)
        group_ne = np.array(group_ne)
        group_dlcfs = np.array(group_dlcfs)
        
        # Build an inner dictionary (per-laser data) for this pulse.
        lasers = {}
        for t_val, rad, te_val, ne_val, dlcfs_val in zip(group_times, group_radius, group_te, group_ne, group_dlcfs):
            key = f"{t_val:.5f}"
            lasers[key] = {
                "time": t_val,
                "radius": rad,
                "te": te_val,
                "ne": ne_val,
                "dlcfs": dlcfs_val
            }
        
        pulse_key = f"Pulse_{i:02d}"  # or use start/end times in the key if preferred.
        pulse_data = {
            "start_time": group_times[0],
            "end_time": group_times[-1],
            "laser_times": group_times,
            "radius_data": group_radius,
            "te_data": group_te,
            "ne_data": group_ne,
            "dlcfs": group_dlcfs,
            "lasers": lasers
        }
        pulses[pulse_key] = pulse_data
    
    # ----------------------------
    # Plotting for Verification
    # ----------------------------
    # Create a directory for plots.
    save_dir = f'graph_{shot}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Example: Plot TS Data (Tₑ vs r) for each pulse (each pulse gets its own figure).
    for pulse_key, pdata in pulses.items():
        plt.figure()
        for laser_key, laser_data in pdata["lasers"].items():
            plt.plot(laser_data["radius"], laser_data["te"], ls='', marker='o', ms=4,
                     label=f't={laser_data["time"]:.5f}s')
        plt.xlabel('r (m)')
        plt.ylabel('T$_e$ (eV)')
        plt.title(f'TS Data for {pulse_key} (t={pdata["start_time"]:.5f}-{pdata["end_time"]:.5f}s)')
        plt.legend(loc='best', fontsize='small')
        plt.grid(True)
        plt.tight_layout()
        fig_path = os.path.join(save_dir, f'figure_{pulse_key}.png')
        plt.savefig(fig_path)
        plt.show()
    
    # (Additional plots—for flattened data, EFIT comparisons, etc.—can be added here as needed.)
    
    # ----------------------------
    # Update Database with All Pulse Data
    # ----------------------------
    pickle_file_path = f'TS_pedestal_{shot}.pkl'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            db = pickle.load(file)
    else:
        db = {}
    
    # For this shot, update its TS pulse data.
    if shot in db:
        shot_data = db[shot]
        shot_data["TS_arrays"] = pulses  # Replace or merge as desired.
    else:
        shot_data = {"TS_arrays": pulses}
    
    db[shot] = shot_data
    update_database(shot, db, pickle_file_path)
    print(f"Data for shot {shot} with {len(pulses)} pulses has been saved to the database.")
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            shots_data = pickle.load(file)
        print("Database loaded successfully.")
    else:
        print("Pickle file not found.")
    
    end_time = datetime.now()
    print("Processing completed. Duration:", end_time - start_time)
    
    results = {
        "pulses": pulses,
        "shots_data": shots_data
    }
    return results

if __name__ == '__main__':
    results_dict = main()
