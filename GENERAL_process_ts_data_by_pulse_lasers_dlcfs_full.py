#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Processing Thomson Scattering Data and Saving Data by User-Specified Pulse

This script retrieves TS laser data, processes and bins the data, generates several
plots, and saves the analyzed data for a specific pulse (defined by a user-specified
time range) into a database. The pulse is stored under a key that includes its start and
end times. In addition, within each pulse the individual laser impulses are stored
in a sub-dictionary (keyed by the laser time) along with their DLCFS, so that later
analysis can be performed either on the full pulse or on a consistent laser channel
across pulses.

Created on Mon Feb  3 16:51:37 2025
@author: qf9176
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
# Main TS Data Processing and Plotting
# =============================================================================
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
    
     # Define the pulse time range you want to analyze.
        #Pulse 1#
#    ts_start = 0.32   # Start TS time [s]
#    ts_end = 0.34  
    #Pulse 2#
    ts_start = 0.82   # Start TS time [s]
    ts_end = 0.84     # End TS time [s]
#    #Pulse 3#
#    ts_start = 0.4       # Start TS time [s] for the pulse
#    ts_end = 0.404       # End TS time [s] for the pulse
#    #Pulse 4#
#    ts_start = 0.432   # Start TS time [s]
#    ts_end = 0.438     # End TS time [s]
#    #Pulse 5#
#    ts_start = 0.5   # Start TS time [s]
#    ts_end = 0.505     # End TS time [s]
##    #Pulse 6#
#    ts_start = 0.53   # Start TS time [s]
#    ts_end = 0.54     # End TS time [s]
##    #Pulse 7#
#    ts_start = 0.56   # Start TS time [s]
#    ts_end = 0.58     # End TS time [s]
    # Define the expected number of laser impulses in this pulse
    num_ts_points = 5
    
    # Generate an array of TS times (uniform grid) for detailed analysis within the chosen pulse range
    ts_times = np.linspace(ts_start, ts_end, num_ts_points)
    
    # Plot style parameters
    ms_size = 4
    m_type = 'o'
    colours = ['red', 'blue', 'green', 'aqua', 'magenta', 'teal', 'orange',
               'brown', 'lime', 'dodgerblue', 'olive', 'gold', 'plum', 'lightcoral',
               'lawngreen', 'darkgreen', 'chocolate', 'tomato', 'steelblue', 'khaki']
    
    # ----------------------------
    # Data Retrieval and Processing
    # ----------------------------
    # Get TS times and corresponding radius data using uniform grid selection
    t_r, d_r, error_code, t_args, ts_actual_times, radius_data = func_get_ts_times_rads(shot, ts_times, t1, t2)
    
    # Sort the TS times and associated arrays in chronological order
    sort_idx = np.argsort(ts_actual_times)
    ts_actual_times = ts_actual_times[sort_idx]
    radius_data = radius_data[sort_idx, :]
    t_args = t_args[sort_idx]
    
    # Retrieve TS time range data (for Tₑ and nₑ channels) for this pulse range
    ts_times_2, radius_data_2, te_data_2, ne_data_2 = func_get_ts_times_rads_range(shot, ts_start, ts_end, t1, t2)
    
    # Flatten 2D arrays for binning and ordering (overall TS data)
    radius_data_3 = np.ravel(radius_data_2)
    te_data_3 = np.ravel(te_data_2)
    ne_data_3 = np.ravel(ne_data_2)
    r_order = np.argsort(radius_data_3)
    
    # Retrieve Tₑ data using the sorted indices (for the uniform grid data)
    trace_name_te = '/AYC/T_E'
    t_te, d_te, error_code_te = get_data_trace(trace_name_te, shot, t1, t2)
    if error_code_te == 0:
        num_tes = d_te.shape[1]
        te_data_specific = np.zeros((len(ts_times), num_tes))
        for z in range(len(ts_times)):
            row_num = int(t_args[z])
            te_data_specific[z, :] = d_te[row_num, :]
    else:
        te_data_specific = np.zeros_like(radius_data)
    
    # Retrieve TS density data from '/AYC/N_E'
    trace_name_ne = '/AYC/N_E'
    t_ne, d_ne, error_code_ne = get_data_trace(trace_name_ne, shot, t1, t2)
    if error_code_ne == 0:
        num_ne = d_ne.shape[1]
        ne_data_specific = np.zeros((len(ts_times), num_ne))
        for z in range(len(ts_times)):
            row_num = int(t_args[z])
            ne_data_specific[z, :] = d_ne[row_num, :]
    else:
        ne_data_specific = np.zeros_like(radius_data)
    
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
    ts_plasma_r = np.interp(ts_times_2, t_efr, efr_dat)
    
    # Compute DLCFS data for the overall TS time range (if needed)
    ts_dlcfs_data_2 = np.zeros_like(radius_data_2)
    for k in range(ts_plasma_r.shape[0]):
        ts_dlcfs_data_2[k, :] = radius_data_2[k, :] - ts_plasma_r[k]
    dlcfs_4 = np.ravel(ts_dlcfs_data_2)
    d_order = np.argsort(dlcfs_4)
    
    # Compute current density and current using TS data
    j0, i0 = func_curr_density(te_data_2, ne_data_2)
    j0_3 = np.ravel(j0)
    i0_3 = np.ravel(i0)
    
    # Bin the data for Tₑ and nₑ using DLCFS bins
    dlcfs_bins = np.arange(-1.0, 0.05, 0.01)
    ne_bins_info = func_bin(dlcfs_4, ne_data_3, dlcfs_bins)
    te_bins_info = func_bin(dlcfs_4, te_data_3, dlcfs_bins)
    
    # ----------------------------
    # Plotting
    # ----------------------------
    save_dir = f'graph_{shot}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Figure 1: TS Data (Tₑ vs r) for each TS time slice in the pulse
    plt.figure(1)
    for z in range(len(ts_actual_times)):
        plt.plot(radius_data[z, :], te_data_specific[z, :],
                 ls='', marker=m_type, ms=ms_size, color=colours[z % len(colours)],
                 label=f't={ts_actual_times[z]:.4f}s')
    plt.xlabel('r (m)')
    plt.ylabel('T$_e$ (eV)')
    plt.title(f'TS Data for Pulse: {ts_start:.3f}-{ts_end:.3f} s')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, f'figure1_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig1_path)
    plt.show()
    
    # Figure 2: Flattened TS Data (Tₑ vs r) sorted by radius for the pulse
    plt.figure(2)
    plt.plot(radius_data_3[r_order], te_data_3[r_order],
             ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'TS time range: {ts_actual_times[0]:.4f} - {ts_actual_times[-1]:.4f} s')
    plt.xlabel('r (m)')
    plt.ylabel('T$_e$ (eV)')
    plt.title(f'Flattened TS Data for Pulse: {ts_start:.3f}-{ts_end:.3f} s')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig2_path = os.path.join(save_dir, f'figure2_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig2_path)
    plt.show()
    
    # Figure 3: EFIT Plasma Radius and TS Acquisition Times
    plt.figure(3)
    plt.plot(t_efr, efr_dat, ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'Shot {shot}')
    plt.plot(ts_times_2, ts_plasma_r, ls='', marker='x', ms=ms_size, color='red',
             label='TS times')
    plt.xlabel('t (s)')
    plt.ylabel('r (m)')
    plt.title('EFIT Plasma Radius and TS Acquisition Times')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig3_path = os.path.join(save_dir, f'figure3_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig3_path)
    plt.show()
    
    # Figure 4: Tₑ vs DLCFS with Binned Data for the pulse
    plt.figure(4)
    plt.plot(dlcfs_4[d_order], te_data_3[d_order],
             ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'Shot {shot}')
    plt.errorbar(dlcfs_bins, te_bins_info[:, 1], te_bins_info[:, 2],
                 ls='-', marker='o', ms=ms_size, color=colours[1],
                 label='Binned')
    plt.xlabel('DLCFS (m)')
    plt.ylabel('T$_e$ (eV)')
    plt.title('T$_e$ vs DLCFS for the Pulse')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig4_path = os.path.join(save_dir, f'figure4_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig4_path)
    plt.show()
    
    # Figure 5: nₑ vs DLCFS with Binned Data for the pulse
    plt.figure(5)
    plt.plot(dlcfs_4[d_order], ne_data_3[d_order],
             ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'Shot {shot}')
    plt.errorbar(dlcfs_bins, ne_bins_info[:, 1], ne_bins_info[:, 2],
                 ls='-', marker='o', ms=ms_size, color=colours[1],
                 label='Binned')
    plt.xlabel('DLCFS (m)')
    plt.ylabel('n$_e$')
    plt.title('n$_e$ vs DLCFS for the Pulse')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig5_path = os.path.join(save_dir, f'figure5_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig5_path)
    plt.show()
    
    # Figure 6: I₀ vs DLCFS for the pulse
    plt.figure(6)
    plt.plot(dlcfs_4[d_order], i0_3[d_order],
             ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'Shot {shot}')
    plt.xlabel('DLCFS (m)')
    plt.ylabel('I$_{0}$ (A)')
    plt.title('I$_{0}$ vs DLCFS for the Pulse')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig6_path = os.path.join(save_dir, f'figure6_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig6_path)
    plt.show()
    
    # Figure 7: TS Density Data at Specific Times for the pulse
    plt.figure(7)
    for z in range(len(ts_actual_times)):
        plt.plot(radius_data[z, :], ne_data_specific[z, :],
                 ls='', marker=m_type, ms=ms_size, color=colours[z % len(colours)],
                 label=f't={ts_actual_times[z]:.4f}s')
    plt.xlabel('r (m)')
    plt.ylabel('n$_e$')
    plt.title('TS Density Data at Specific Times for the Pulse')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig7_path = os.path.join(save_dir, f'figure7_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig7_path)
    plt.show()
    
    # Figure 8: Flattened TS Density Data (nₑ vs r) sorted by radius for the pulse
    plt.figure(8)
    plt.plot(radius_data_3[r_order], ne_data_3[r_order],
             ls='', marker=m_type, ms=ms_size, color=colours[0],
             label=f'TS time range: {ts_actual_times[0]:.4f} - {ts_actual_times[-1]:.4f} s')
    plt.xlabel('r (m)')
    plt.ylabel('n$_e$')
    plt.title('Flattened TS Density Data for the Pulse')
    plt.legend(loc='best', fontsize='small')
    plt.grid()
    plt.tight_layout()
    fig8_path = os.path.join(save_dir, f'figure8_{ts_start:.3f}-{ts_end:.3f}.png')
    plt.savefig(fig8_path)
    plt.show()
    
    # ----------------------------
    # Update Database with Pulse Data
    # ----------------------------
    pickle_file_path = f'TS_pedestal_{shot}.pkl'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            db = pickle.load(file)
    else:
        db = {}
    
    # If the shot already exists, retrieve its data; otherwise create a new entry.
    if shot in db:
        shot_data = db[shot]
        if "Te_arrays" not in shot_data:
            shot_data["Te_arrays"] = {}
    else:
        shot_data = {"Te_arrays": {}}
    
    # Create a unique key for the pulse using the specified start and end times.
    pulse_key = f'Pulse_{ts_start:.3f}_{ts_end:.3f}'
    
    # Build an inner dictionary with each laser (TS time) data.
    # For each laser, we compute DLCFS by interpolating the plasma radius at its time.
    lasers = {}
    for i, t_val in enumerate(ts_actual_times):
        # Interpolate plasma radius at this laser's time.
        plasma_radius = np.interp(t_val, t_efr, efr_dat)
        # Compute DLCFS for this laser.
        laser_dlcfs = radius_data[i, :] - plasma_radius
        key = f"{t_val:.5f}"
        lasers[key] = {
            "time": t_val,
            "radius": radius_data[i, :],
            "te": te_data_specific[i, :],
            "ne": ne_data_specific[i, :],
            "dlcfs": laser_dlcfs
        }
    
    # Build the pulse_data dictionary.
    pulse_data = {
        "start_time": ts_start,
        "end_time": ts_end,
        "ts_actual_times": ts_actual_times,
        "radius_data": radius_data,
        "te_data_specific": te_data_specific,
        "ne_data_specific": ne_data_specific,
        "lasers": lasers
    }
    
    # Optionally, also store overall (flattened) data.
    shot_data["dlcsf_te"] = dlcfs_4[d_order]
    shot_data["te_data"] = te_data_3[r_order]
    shot_data["dlcs_bins"] = dlcfs_bins
    shot_data["te_bins"] = te_bins_info[:, 1]
    shot_data["te_bins_error"] = te_bins_info[:, 2]
    shot_data["ne_data"] = ne_data_3[d_order]
    shot_data["ne_bins"] = ne_bins_info[:, 1]
    shot_data["ne_bins_error"] = ne_bins_info[:, 2]
    shot_data["laser_time"] = ts_actual_times
    
    # Save the pulse data into the shot's "Te_arrays" dictionary.
    shot_data["Te_arrays"][pulse_key] = pulse_data
    
    db[shot] = shot_data
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(db, file)
    print(f"Data for shot {shot}, pulse {pulse_key} has been saved to the database.")
    
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as file:
            shots_data = pickle.load(file)
        print("Database loaded successfully.")
    else:
        print("Pickle file not found.")
    
    end_time = datetime.now()
    print("Processing completed. Duration:", end_time - start_time)
    
    results = {
        "ts_actual_times": ts_actual_times,
        "radius_data": radius_data,
        "te_data_specific": te_data_specific,
        "ne_data_specific": ne_data_specific,
        "shots_data": shots_data,
        "Te_arrays": shot_data["Te_arrays"]
    }
    return results

if __name__ == '__main__':
    results_dict = main()
