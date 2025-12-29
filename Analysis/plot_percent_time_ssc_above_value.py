#################### Percent of Time SSC in Top 1 Meter Above Threshold ##########################
# The purpose of this script is to plot the percent of time that the SSC in the top 1 m
# of the water column is above a certain user-specified threshold(s)

# Notes:
# - This needs to be run in xroms environment 
#
####################################################################################################


# Load in the packages 
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import xroms 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import cmocean 
import scipy.io
from glob import glob
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import colors


# Set a universal fontsize
fontsize = 20

# Set the tick size for all plots
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize)

# Prevent tick labels from overlapping
matplotlib.rcParams['xtick.major.pad'] = 10
matplotlib.rcParams['ytick.major.pad'] = 10


# Load in the grid
grid = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Include/KakAKgrd_shelf_big010_smooth006.nc')
#grid = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Grids/KakAKgrd_shelf_big010_smooth006.nc')

# Pull out some dimensions
eta_rho_len = len(grid.eta_rho)
xi_rho_len = len(grid.xi_rho)

# Load in the rho masks
mask_rho_nan = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_ones_nans.nc') # UPDATE PATH
mask_rho_zeros = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_zeros_ones.nc')
#mask_rho_nan = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_ones_nans.nc')
#mask_rho_zeros = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_zeros_ones.nc')


# ---------------------------------------------------------------------------
# ---------------------- Define Functions -----------------------------------
# ---------------------------------------------------------------------------

# Define a function to pull out the lenght of time in the model run
# And the time steps
def get_model_time(filenames, num_files):
    """
    This function loops though model output and pulls
    out the entire length of the run, as well as the 
    individual time steps of the run.
    
    Inputs:
    - filenames: path and name of model output
    - num_files: the number of model output files
    
    Outputs:
    - time_len: length of time of model run (integer)
    - time_steps_list: list of time steps of full run (datetimes64)
    - time_lengths: array holding the lenght of time of each output file
    """

    # Create an array to hold the length of time in each output file
    time_lengths = np.empty((num_files))

    # Loop through output to pull out lenth of time
    for k in range(num_files):
        # Open the output file
        model_output = xr.open_dataset(filenames[k])

        # Pull out the length of time 
        time_lengths[k] = len(model_output.ocean_time)

    # Now sum over all the lengths to get total time
    time_len = np.sum(time_lengths, axis=0)

    # Convert from float to int
    time_len = int(time_len)

    # Loop back through the output to pull out the time step and save it
    # Make a list to hold the time steps 
    time_steps_list = []
    # Loop through output
    for h in range(num_files):
        # Open the output file
        model_output = xr.open_dataset(filenames[h])

        # Get the length of the run
        output_len = len(model_output.ocean_time)

        # Loop through each time step and append it to the list
        for g in range(output_len):
            time_steps_list.append(model_output.ocean_time[g].values)

    # Return this time length and time steps
    return(time_len, time_steps_list, time_lengths)


# Make a  function to interpolate the ROMS data to a given depth (measured from
# top down and negative)
def interp_roms_ssc_from_top(filenames, time_len, eta_rho_len, xi_rho_len, time_lengths, num_files, depths, threshold_01, threshold_02, threshold_03):
    """
    This function takes all given ROMS ocean_his files, opens one at a time and 
    interpolates total SSC onto the given depths, then returns the interpolated
    data. Right now, it is set up to interpolate total SSC.
    
    Inputs:
    - filenames: path and name of model output
    - time_len: full length of time of all model outputs combined 
    - eta_rho_len: length of eta dimension
    - xi_rho_len: length of xi dimension
    - time_lengths: lengths of time for each model output
    - num_files: the number of model output files
    - depths: the depths to interpolate the data onto 
    - threshold: SSC threshold to see how often the data is above (kg/m3)
    Outputs:
    - tot_ssc_top_1m: Time series of total SSC 1 m above the seafloor (over all space)
    

    Returns
    -------
    None.

    """
    # Make empty arrays to hold the data
    tot_ssc_top_1m = np.empty((time_len, len(depths), eta_rho_len, xi_rho_len))
    avg_tot_ssc_top_1m = np.empty((time_len, eta_rho_len, xi_rho_len))
    time_above_threshold_01_cnt = np.empty((num_files, eta_rho_len, xi_rho_len))
    time_above_threshold_02_cnt = np.empty((num_files, eta_rho_len, xi_rho_len))
    time_above_threshold_03_cnt = np.empty((num_files, eta_rho_len, xi_rho_len))
    
    # Set the initial time step
    time_step = 0
    
    # Loop through each output file
    for jj in range(num_files):
        print('jj for top 1 m:' , jj, flush=True)
        # Load in the ROMS output 
        ds = xr.open_dataset(filenames[jj])
        #ds = xr.open_dataset(filenames)
        ds['h'] = ds.bath
        ds, xgrid = xroms.roms_dataset(ds)
    
        # Get the depths as height from seabed
        height_from_seabed = ds.z_rho + ds.bath
        height_from_seabed.name = 'z_rho'
    
        # Set two depths, 1 m above seafloor
        #depths = np.asarray([1.0])

        # Interopolate onto the given CODA depths
        ssc_allsed_interp_top_1m = xroms.isoslice((ds.sand_01+ds.sand_02+ds.sand_03+ds.mud_01+ds.mud_02+ds.mud_03+ds.mud_04+ds.mud_05+ds.mud_06+
                                                   ds.mud_07+ds.mud_08+ds.mud_09+ds.mud_10+ds.mud_11+ds.mud_12+ds.mud_13+ds.mud_14+ds.mud_15+ds.mud_16+
                                                   ds.mud_17+ds.mud_18+ds.mud_19+ds.mud_20+ds.mud_21+ds.mud_22+ds.mud_23+ds.mud_24), depths, xgrid)
    
        
        # Take the average over depth
        #tot_ssc_top_1m_avg = np.nanmean(ssc_allsed_interp_top_1m, axis=1)
        tot_ssc_top_1m_avg = ssc_allsed_interp_top_1m.mean(axis=1, skipna=True)
        
        # Get the number of times steps this is above a threshold
        time_above_threshold_01_cnt_tmp = (tot_ssc_top_1m_avg.where(tot_ssc_top_1m_avg > threshold_01).groupby('eta_rho',).count(dim='ocean_time'))
        time_above_threshold_02_cnt_tmp = (tot_ssc_top_1m_avg.where(tot_ssc_top_1m_avg > threshold_02).groupby('eta_rho',).count(dim='ocean_time'))
        time_above_threshold_03_cnt_tmp = (tot_ssc_top_1m_avg.where(tot_ssc_top_1m_avg > threshold_03).groupby('eta_rho',).count(dim='ocean_time'))
        
        # Save these to the arrays 
        #print('time_step: ', time_step)
        #print('time_step + time_lengths[j]: ', time_step+time_lengths[j])
        start = int(time_step)
        end = int(time_step+time_lengths[jj])
        #end = int(time_step+time_lengths)
        tot_ssc_top_1m[start:end,:,:,:] = ssc_allsed_interp_top_1m
        avg_tot_ssc_top_1m[start:end,:,:] = tot_ssc_top_1m_avg
        time_above_threshold_01_cnt[jj,:,:] = time_above_threshold_01_cnt_tmp
        time_above_threshold_02_cnt[jj,:,:] = time_above_threshold_02_cnt_tmp
        time_above_threshold_03_cnt[jj,:,:] = time_above_threshold_03_cnt_tmp
        
        # Update the base time_step
        time_step = time_step + time_lengths[jj]
        #time_step = time_step + time_lengths

    # Now combine all of the counts to get total counts/all time
    tot_cnt_time_above_threshold_01 = np.sum(time_above_threshold_01_cnt, axis=0)
    tot_cnt_time_above_threshold_02 = np.sum(time_above_threshold_02_cnt, axis=0)
    tot_cnt_time_above_threshold_03 = np.sum(time_above_threshold_03_cnt, axis=0)
     
    # Return the ssc
    return(tot_ssc_top_1m, avg_tot_ssc_top_1m, tot_cnt_time_above_threshold_01, tot_cnt_time_above_threshold_02, tot_cnt_time_above_threshold_03)
    #return(tot_ssc_top_1m, avg_tot_ssc_top_1m, time_above_threshold_01_cnt_tmp)


# ---------------------------------------------------------------------------
# ---------------------- Process the Data -----------------------------------
# ---------------------------------------------------------------------------

# Want a time series of average SSC in top 1 m over all space and time 
# Loop through model output and call the function
# First, get all the file names 
# For now, use the run where the unique class is unaggregated mud but might 
# want to make version with aggregated mud, too, so see if the results are different 
# although they definitely shouldn't be

# --- Aggregated ---
# ROMS aggregated output
# dbsed0001
file_names_agg = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_002_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')

# Sort them to be in order
file_names_agg2 = sorted(file_names_agg)

# Check to see if this worked
print(file_names_agg2[0], flush=True)
print(file_names_agg2[1], flush=True)
print(file_names_agg2[2], flush=True)
print(file_names_agg2[-1], flush=True)

# Pull out the number of files
num_files_agg = len(file_names_agg2)

# Pull out the length of time of the full run, the time steps, 
# and the length of time of each output file
# Aggregated 
full_time_len_agg, time_steps_agg, time_lengths_agg = get_model_time(file_names_agg2, num_files_agg)

# --- Unaggregated ---
# Same but for unagregated
# ROMS aggregated output
# dbsed0001
file_names_unag = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_003_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc')

# Sort them to be in order
file_names_unag2 = sorted(file_names_unag)

# Check to see if this worked
print(file_names_unag2[0], flush=True)
print(file_names_unag2[1], flush=True)
print(file_names_unag2[2], flush=True)
print(file_names_unag2[-1], flush=True)

# Pull out the number of files
num_files_unag = len(file_names_unag2)

# Pull out the length of time of the full run, the time steps, 
# and the length of time of each output file
# Unaggregated 
full_time_len_unag, time_steps_unag, time_lengths_unag = get_model_time(file_names_unag2, num_files_unag)


# ------ SSC ------
# Interpolate it to the desired depths and determine the number
# of tme steps where it is above a given SSC threshold

# Set the SSC threshold
threshold_01 = 0.001 # kg/m3
threshold_02 = 0.01 # kg/m3
threshold_03 = 0.05 # kg/m3

# 1 m above seafloor
#tot_ssc_bot_1m, tot_ssc_og = interp_roms_ssc_to1m_rho(file_names2, full_time_len, s_rho_len, eta_rho_len, xi_rho_len, time_lengths, num_files)

# 1 m below surface
#depth_surf = np.asarray([-1])
#depths_surf = np.asarray([-0.5,-1])
depths_surf = np.asarray([-0.25,-0.75])
#tot_ssc_top_1m_agg = interp_roms_ssc_from_top(file_names_agg2, full_time_len_agg, eta_rho_len, xi_rho_len, time_lengths_agg, num_files_agg, depths_surf)  #OLD
# NEW USE THIS
# Aggregated 
#tot_ssc_top_1m_agg, avg_tot_ssc_top_1m_agg, tot_cnt_time_above_threshold_01_agg = interp_roms_ssc_from_top(file_names_agg2, full_time_len_agg, eta_rho_len, xi_rho_len, time_lengths_agg, num_files_agg, depths_surf, threshold_01)
tot_ssc_top_1m_agg, avg_tot_ssc_top_1m_agg, tot_cnt_time_above_threshold_01_agg, tot_cnt_time_above_threshold_02_agg, tot_cnt_time_above_threshold_03_agg = interp_roms_ssc_from_top(file_names_agg2, full_time_len_agg, eta_rho_len, xi_rho_len, time_lengths_agg, num_files_agg, depths_surf, threshold_01, threshold_02, threshold_03)
# Unaggregated 
#tot_ssc_top_1m_unag, avg_tot_ssc_top_1m_unag, tot_cnt_time_above_threshold_01_unag, tot_cnt_time_above_threshold_02_unag, tot_cnt_time_above_threshold_03_unag = interp_roms_ssc_from_top(file_names_unag2, full_time_len_unag, eta_rho_len, xi_rho_len, time_lengths_unag, num_files_unag, depths_surf, threshold_01, threshold_02, threshold_03)

# Print a bunch of things to see if this worked okay 
# Aggregated 
print('tot_ssc_top_1m_agg  shape: ', np.shape(tot_ssc_top_1m_agg), flush=True)
print('avg_tot_ssc_top_1m_agg shape: ', np.shape(avg_tot_ssc_top_1m_agg), flush=True)
print('time_above_threshold_01_cnt_tmp: ', np.shape(tot_cnt_time_above_threshold_01_agg), flush=True)
# Unaggregated 
# print('tot_ssc_top_1m_unag  shape: ', np.shape(tot_ssc_top_1m_unag), flush=True)
# print('avg_tot_ssc_top_1m_unag shape: ', np.shape(avg_tot_ssc_top_1m_unag), flush=True)
# print('time_above_threshold_01_cnt_tmp: ', np.shape(tot_cnt_time_above_threshold_01_unag), flush=True)

# Calculate the percent of time from the counts 
# Aggreagted 
percent_time_above_threshold_01_agg = (tot_cnt_time_above_threshold_01_agg/full_time_len_agg)
percent_time_above_threshold_02_agg = (tot_cnt_time_above_threshold_02_agg/full_time_len_agg)
percent_time_above_threshold_03_agg = (tot_cnt_time_above_threshold_03_agg/full_time_len_agg)
# Unaggregated
# percent_time_above_threshold_01_unag = (tot_cnt_time_above_threshold_01_unag/full_time_len_unag)
# percent_time_above_threshold_02_unag = (tot_cnt_time_above_threshold_02_unag/full_time_len_unag)
# percent_time_above_threshold_03_unag = (tot_cnt_time_above_threshold_03_unag/full_time_len_unag)


# ---------------------------------------------------------------------------
# ---------------------- Save to a NetCDF -----------------------------------
# ---------------------------------------------------------------------------
# Assuming this all works, save the post-processed data to
# a netCDF just in case 

# Set up the data
roms_ssc_time_above_threshold = xr.Dataset(
    data_vars=dict(
        time_above_threshold_01=(['eta_rho','xi_rho'], percent_time_above_threshold_01_agg),
        time_above_threshold_02=(['eta_rho','xi_rho'], percent_time_above_threshold_02_agg),
        time_above_threshold_03=(['eta_rho','xi_rho'], percent_time_above_threshold_03_agg)
        ),
    coords=dict(
        xi_rho=('xi_rho', grid.xi_rho.values),
        eta_rho=('eta_rho', grid.eta_rho.values), 
        ),
    attrs=dict(description='Time above SSC three thresholds (0.001, 0.01, 0.05 kg/m3) for the aggregated model runs')) 
# Add more metadata?
roms_ssc_time_above_threshold.time_above_threshold_01.name='fraction of time above three SSC thresholds (0.001 kg/m3) in aggregated mud run for average total SSC in top 1 m'
roms_ssc_time_above_threshold.time_above_threshold_02.name='fraction of time above three SSC thresholds (0.01 kg/m3) in aggregated mud run for average total SSC in top 1 m'
roms_ssc_time_above_threshold.time_above_threshold_03.name='fraction of time above three SSC thresholds (0.05 kg/m3) in aggregated mud run for average total SSC in top 1 m'

# Save to a netcdf
roms_ssc_time_above_threshold.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/time_ssc_top_1m_above_three_thresholds_agg.nc')

print('Done saving to NetCDF', flush=True)

# Print things to see how this worked
# Aggregated 
print('time_above_threshold_01 shape: ', np.shape(percent_time_above_threshold_01_agg), flush=True)
print('percent_time_above_threshold_01_agg min: ', np.min(percent_time_above_threshold_01_agg), flush=True)
print('percent_time_above_threshold_01_agg max: ', np.max(percent_time_above_threshold_01_agg), flush=True)
print('percent_time_above_threshold_01_agg mean: ', np.mean(percent_time_above_threshold_01_agg), flush=True)
# Unaggreagted 
# print('time_above_threshold_01 shape: ', np.shape(percent_time_above_threshold_01_unag), flush=True)
# print('percent_time_above_threshold_01_unag min: ', np.min(percent_time_above_threshold_01_unag), flush=True)
# print('percent_time_above_threshold_01_unag max: ', np.max(percent_time_above_threshold_01_unag), flush=True)
# print('percent_time_above_threshold_01_unag mean: ', np.mean(percent_time_above_threshold_01_unag), flush=True)

