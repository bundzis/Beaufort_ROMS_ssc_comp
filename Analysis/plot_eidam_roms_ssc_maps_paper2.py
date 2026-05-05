#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:37:58 2023

@author: brun1463
"""

########################### Eidam ROMS SSC Comparison Maps ######################
# The purpose of this script is to make plots comparing the ROMS SSC values 
# with the Eidam CTD observations. This will be don eby plotting the values 
# 1 m below the ocean surface and 1 m above the seafloor for both ROMS 
# and the observations and plotting them as a map. 
#
# Notes:
# - This is currently setup to only use the observations from 2020
# - At some point, need to filter out values that are greater than 35 in NTU
# - This script has been updated to use the 2020 ROMS output
# - This is the same as the script used in paper 1 but edited to be fore paper 2
#################################################################################


# Load in the packages 
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import xroms 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pylab as pl
import cmocean 
import scipy.io
from datetime import datetime #as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import pearsonr
import scipy.io
from datetime import datetime #as dt
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
crs = ccrs.PlateCarree()

# Set a universal fontsize
fontsize = 20

# Set the tick size for all plots
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize)

# Prevent tick labels from overlapping
matplotlib.rcParams['xtick.major.pad'] = 10
matplotlib.rcParams['ytick.major.pad'] = 10


# Load in the grid
#grid = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Include/KakAKgrd_shelf_big010_smooth006.nc')
grid = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Grids/KakAKgrd_shelf_big010_smooth006.nc') # UPDATE PATH

# Pull out some dimensions
eta_rho_len = len(grid.eta_rho)
xi_rho_len = len(grid.xi_rho)
s_rho_len = int(20)

# Load in the rho masks 
mask_rho_nan = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_ones_nans.nc')
mask_rho_zeros = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_zeros_ones.nc')


# Load in the Eidam data
#turbidity_data_2019 = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Eidam_data/BeaufortShelf_CTDTu_2019.nc', decode_times=False)
turbidity_data_2020 = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Eidam_data/BeaufortShelf_CTDTu_2020.nc', decode_times=False)
#turbidity_data_2021 = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Eidam_data/BeaufortShelf_CTDTuPARchlKd_2021.nc', decode_times=False)
#turbidity_data_2022 = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Eidam_data/BeaufortShelf_CTDTuPARchlKd_2022.nc', decode_times=False)


# Look at th data to get the times (also subtract 9 hours to go from UTC to local Alaska time)
#turbidity_data_2019['datetime'] = pd.to_datetime(turbidity_data_2019.time.values-719528-(9/24), unit='d')
turbidity_data_2020_datetime = pd.to_datetime(turbidity_data_2020.time.values-719528-(9/24), unit='d')
#turbidity_data_2021_datetime = pd.to_datetime(turbidity_data_2021.time.values-719528-(9/24), unit='d')
#turbidity_data_2022_datetime = pd.to_datetime(turbidity_data_2022.time.values-719528-(9/24), unit='d')


# Make a bunch of functions for processing the data
# Make a function to interpoalte the ROMS ssc to 1 m above the saebed
def interp_roms_ssc_to1m_rho(filename):
    """
    This function takes a given ROMS ocean_his file, opens it and 
    interpolates it onto the given depths, then returns the interpolated
    data. Right now, it is set up to interpolate u and v currents.
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)
    
    # Get the depths as height from seabed
    height_from_seabed = ds.z_rho + ds.bath
    height_from_seabed.name = 'z_rho'
    
    # Set two depths, 1 m above seafloor
    depths = np.asarray([1.0])

    # Interopolate onto the given CODA depths
    #ssc_allsed_interp_1m = xroms.isoslice((ds.mud_01+ds.mud_02+ds.sand_01+ds.sand_02+ds.sand_03), depths, xgrid, 
                                         #iso_array=height_from_seabed, axis='Z')
    ssc_allsed_interp_1m = xroms.isoslice((ds.sand_01+ds.sand_02+ds.sand_03+ds.mud_01+ds.mud_02+ds.mud_03+
                                           ds.mud_04+ds.mud_05+ds.mud_06+ds.mud_07+ds.mud_08+ds.mud_09+ds.mud_10+
                                           ds.mud_11+ds.mud_12+ds.mud_13+ds.mud_14+ds.mud_15+ds.mud_16+ds.mud_17+
                                           ds.mud_18+ds.mud_19+ds.mud_20+ds.mud_21+ds.mud_22+ds.mud_23+ds.mud_24), depths, xgrid, 
                                         iso_array=height_from_seabed, axis='Z')
     
    # Return the ssc
    return(ssc_allsed_interp_1m)


# Make a  function to interpolate the ROMS data to a given depth (measured from
# top down and negative)
def interp_roms_ssc_output_fromtop(filename, depths):
    """
    This function takes a given ROMS ocean_his file and 
    interpolates it onto the given depths, then returns the interpolated
    data. 
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)

    
    # Interopolate onto the given CODA depths
    #ssc_roms_interp_depths = xroms.isoslice((ds.mud_01+ds.mud_02+ds.sand_01+ds.sand_02+ds.sand_03), 
     #                                       depths, xgrid)
    ssc_roms_interp_depths = xroms.isoslice((ds.sand_01+ds.sand_02+ds.sand_03+ds.mud_01+ds.mud_02+ds.mud_03+
                                           ds.mud_04+ds.mud_05+ds.mud_06+ds.mud_07+ds.mud_08+ds.mud_09+ds.mud_10+
                                           ds.mud_11+ds.mud_12+ds.mud_13+ds.mud_14+ds.mud_15+ds.mud_16+ds.mud_17+
                                           ds.mud_18+ds.mud_19+ds.mud_20+ds.mud_21+ds.mud_22+ds.mud_23+ds.mud_24), 
                                             depths, xgrid)
    
    # Return these currents 
    return(ssc_roms_interp_depths)


# Make a function to interpoalte the ROMS temperature to 1 m above the saebed
def interp_roms_temp_to1m_rho(filename):
    """
    This function takes a given ROMS ocean_his file, opens it and 
    interpolates it onto the given depths, then returns the interpolated
    data. Right now, it is set up to interpolate temperature.
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)
    
    # Get the depths as height from seabed
    height_from_seabed = ds.z_rho + ds.bath
    height_from_seabed.name = 'z_rho'
    
    # Set two depths, 1 m above seafloor
    depths = np.asarray([1.0])

    # Interopolate onto the given CODA depths
    temp_allsed_interp_1m = xroms.isoslice((ds.temp), depths, xgrid, 
                                         iso_array=height_from_seabed, axis='Z')
     
    # Return the ssc
    return(temp_allsed_interp_1m)


# Make a  function to interpolate the ROMS data to a given depth (measured from
# top down and negative)
def interp_roms_temp_output_fromtop(filename, depths):
    """
    This function takes a given ROMS ocean_his file and 
    interpolates it onto the given depths, then returns the interpolated
    data. 
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)

    
    # Interopolate onto the given CODA depths
    temp_roms_interp_depths = xroms.isoslice((ds.temp), 
                                            depths, xgrid)
    
    # Return these currents 
    return(temp_roms_interp_depths)


# Make a function to interpoalte the ROMS salinity to 1 m above the saebed
def interp_roms_salt_to1m_rho(filename):
    """
    This function takes a given ROMS ocean_his file, opens it and 
    interpolates it onto the given depths, then returns the interpolated
    data. Right now, it is set up to interpolate salinity.
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)
    
    # Get the depths as height from seabed
    height_from_seabed = ds.z_rho + ds.bath
    height_from_seabed.name = 'z_rho'
    
    # Set two depths, 1 m above seafloor
    depths = np.asarray([1.0])

    # Interopolate onto the given CODA depths
    salt_allsed_interp_1m = xroms.isoslice((ds.salt), depths, xgrid, 
                                         iso_array=height_from_seabed, axis='Z')
     
    # Return the ssc
    return(salt_allsed_interp_1m)


# Make a  function to interpolate the ROMS data to a given depth (measured from
# top down and negative)
def interp_roms_salt_output_fromtop(filename, depths):
    """
    This function takes a given ROMS ocean_his file and 
    interpolates it onto the given depths, then returns the interpolated
    data. 
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)

    
    # Interopolate onto the given CODA depths
    salt_roms_interp_depths = xroms.isoslice((ds.salt), 
                                            depths, xgrid)
    
    # Return these currents 
    return(salt_roms_interp_depths)


# Make a function to interpoalte the ROMS salinity to 1 m above the seabed
def interp_roms_rhoanom_to1m_rho(filename):
    """
    This function takes a given ROMS ocean_his file, opens it and 
    interpolates it onto the given depths, then returns the interpolated
    data. Right now, it is set up to interpolate salinity.
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)
    
    # Get the depths as height from seabed
    height_from_seabed = ds.z_rho + ds.bath
    height_from_seabed.name = 'z_rho'
    
    # Set two depths, 1 m above seafloor
    depths = np.asarray([1.0])

    # Interopolate onto the given CODA depths
    rhoanom_allsed_interp_1m = xroms.isoslice((ds.rho), depths, xgrid, 
                                         iso_array=height_from_seabed, axis='Z')
     
    # Return the ssc
    return(rhoanom_allsed_interp_1m)


# Make a  function to interpolate the ROMS data to a given depth (measured from
# top down and negative)
def interp_roms_rhoanom_output_fromtop(filename, depths):
    """
    This function takes a given ROMS ocean_his file and 
    interpolates it onto the given depths, then returns the interpolated
    data. 
    

    Returns
    -------
    None.

    """
    
    # Load in the ROMS output 
    ds = xr.open_dataset(filename)
    ds['h'] = ds.bath
    ds, xgrid = xroms.roms_dataset(ds)

    
    # Interopolate onto the given CODA depths
    salt_roms_interp_depths = xroms.isoslice((ds.rho), 
                                            depths, xgrid)
    
    # Return these currents 
    return(salt_roms_interp_depths)


# Make a function to interpolate the observations to 1 m above their bottom 
# depth

# Make a function to interpolate the observations to a given depth (measured 
# from top down)



# ------------------------------ Prep ROMS -------------------------------------
# Load in the associated ROMS output for 2020 ROMS version
# dbsed0003
# Paper 2 Aggregated
#filename5 = '/Volumes/Model_Output3/Paper2/Aggregated/beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_00013.nc'
# Paper 2 Unaggregated
filename5 = '/Volumes/Model_Output3/Paper2/Unaggregated/beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_00013.nc'


begin1 = '2020-09-26'
end1 = '2020-09-30'

# ------ SSC ------
# Interpolate it to the desired depths 
# 1 m above seafloor
roms_ssc_1m_above_seafloor = interp_roms_ssc_to1m_rho(filename5)

# 1 m below surface
depth_surf = np.asarray([-1])
roms_ssc_1m_from_surf = interp_roms_ssc_output_fromtop(filename5, depth_surf)

# Slice these to the time period we want 
roms_ssc_1m_above_seafloor_p1 = roms_ssc_1m_above_seafloor.sel(ocean_time=slice(begin1, end1))
roms_ssc_1m_from_surf_p1 = roms_ssc_1m_from_surf.sel(ocean_time=slice(begin1, end1))

# Take the average over time
roms_ssc_1m_above_seafloor_p1_avg = np.mean(roms_ssc_1m_above_seafloor_p1, axis=0)
roms_ssc_1m_from_surf_p1_avg = np.mean(roms_ssc_1m_from_surf_p1, axis=0)

print('done with roms ssc')

# ------ Temperature ------
# Interpolate it to the desired depths 
# 1 m above seafloor
roms_temp_1m_above_seafloor = interp_roms_temp_to1m_rho(filename5)

# 1 m below surface
depth_surf = np.asarray([-1])
roms_temp_1m_from_surf = interp_roms_temp_output_fromtop(filename5, depth_surf)

# Slice these to the time period we want 
roms_temp_1m_above_seafloor_p1 = roms_temp_1m_above_seafloor.sel(ocean_time=slice(begin1, end1))
roms_temp_1m_from_surf_p1 = roms_temp_1m_from_surf.sel(ocean_time=slice(begin1, end1))

# Take the average over time
roms_ptemp_1m_above_seafloor_p1_avg = np.mean(roms_temp_1m_above_seafloor_p1, axis=0)
roms_ptemp_1m_from_surf_p1_avg = np.mean(roms_temp_1m_from_surf_p1, axis=0)

print('done with roms temp')

# ------ Salinity ------
# Interpolate it to the desired depths 
# 1 m above seafloor
roms_salt_1m_above_seafloor = interp_roms_salt_to1m_rho(filename5)

# 1 m below surface
depth_surf = np.asarray([-1])
roms_salt_1m_from_surf = interp_roms_salt_output_fromtop(filename5, depth_surf)

# Slice these to the time period we want 
roms_salt_1m_above_seafloor_p1 = roms_salt_1m_above_seafloor.sel(ocean_time=slice(begin1, end1))
roms_salt_1m_from_surf_p1 = roms_salt_1m_from_surf.sel(ocean_time=slice(begin1, end1))

# Take the average over time
roms_salt_1m_above_seafloor_p1_avg = np.mean(roms_salt_1m_above_seafloor_p1, axis=0)
roms_salt_1m_from_surf_p1_avg = np.mean(roms_salt_1m_from_surf_p1, axis=0)

print('done with roms salt')

# ---------------- Temperature from Potential Temperature ---------------------
# Use the salinity and potential temperature at 1 m below surface and above seafloor 
# from above to calculate temperature 
# Load in the seawater package  
import seawater as sw
# The function is sw.temp(s, pt, p, pr=0)

# Set the reference pressure 
ref_pres = 10.13 # decibar

# Calculate the pressure 
# Set gravity
g = 9.82 # m/s2

# Set rho0 from ROMS 
rho0 = 1025 # kg/m3

# Set the depths 
surf_depth = 1 # m 
bot_depth = (grid.h.values - 1) # m

# Get density anomaly interpolated onto these depths 
# 1 m above seafloor
rho_anom_1m_bot = interp_roms_temp_to1m_rho(filename5)
# 1 m below surface
depth_surf = np.asarray([-1])
rho_anom_1m_surf = interp_roms_temp_output_fromtop(filename5, depth_surf)

# Average these over time 
rho_anom_1m_bot_avg = np.mean(rho_anom_1m_bot, axis=0)
rho_anom_1m_surf_avg = np.mean(rho_anom_1m_surf, axis=0)

# Calculate associated pressures 
surf_1m_pres = (rho_anom_1m_surf_avg + rho0)* g * surf_depth # Pa
bot_1m_pres = (rho_anom_1m_bot_avg + rho0) * g * bot_depth # Pa

# Conver that to decibars 
surf_1m_pres_db = surf_1m_pres/10000
bot_1m_pres_db = bot_1m_pres/10000

# Now give these values to the seawater function 
roms_temp_1m_above_seafloor_p1_avg = sw.temp(roms_salt_1m_above_seafloor_p1_avg, roms_ptemp_1m_above_seafloor_p1_avg, bot_1m_pres_db, ref_pres)
roms_temp_1m_from_surf_p1_avg = sw.temp(roms_salt_1m_from_surf_p1_avg, roms_ptemp_1m_from_surf_p1_avg, surf_1m_pres_db, ref_pres)


# (this will be plotted as a contour in the background so I think we are good to go
# for plotting)




# -----------------------------Prep Observations ---------------------------------
# Before we pull out the values we want, need to go through and find the 
# deepest depth without nan for each location, then subtract 1 m from that 
# to get the depths of 1 m above the seabed for each location 
# Except the dataset uses a fill value instead of nan sooooo maybe replace fill values with nan then loop
turbidity_data_2020_nan = np.where(turbidity_data_2020.turbidity[:,:] >6000, np.nan, turbidity_data_2020.turbidity[:,:])

# Make an array to hold the values
obs_bottom_depths = np.empty((len(turbidity_data_2020.n)))

# Manually set the index of the first nan value for each location
# Idx with nans in column: 9,
bot_idx = np.asarray([68, 60, 55, 40, 37, 29, 27, 21, 17, 27, 30, 34, 44, 37, 30,
                      24, 18, 14, 71, 75, 90, 105, 41, 55, 60, 67, 73, 75, 79, 41,
                      36,35,27,53,65,82,119,49,52,46,35,35,29,27,26,23,15,9,21,25,
                      28,27,10,16,19,23,25,26,57,70,92,53,32,44,38,33,33,33,31,37,
                      38,31,25,23,22,15,13,25,29,29,28,24,17,14,34,35,37,27,42,52,
                      117,189])

# Find these dpeths, then subtract one to get 1 m above seafloor
# Loop through the points
for i in range(len(turbidity_data_2020.n)):
    # Find the bottom depth
    bot_depth = turbidity_data_2020.depths[bot_idx[i],i]
    
    # Subtract 1 
    bot_depth_tmp = bot_depth - 1
    
    # Save this to the array
    obs_bottom_depths[i] = bot_depth_tmp
    
    
# Assuming this worked, loop through all locations and interpolate them to 
# 1 m above the surface and 1 m above the seafloor and save to an array
# Make empty arrays to hold the data 
obs_ssc_1m_above_seafloor_p1 = np.empty(len(turbidity_data_2020.n))
obs_ssc_1m_from_surf_p1 = np.empty(len(turbidity_data_2020.n))
obs_temp_1m_above_seafloor_p1 = np.empty(len(turbidity_data_2020.n))
obs_temp_1m_from_surf_p1 = np.empty(len(turbidity_data_2020.n))
obs_salt_1m_above_seafloor_p1 = np.empty(len(turbidity_data_2020.n))
obs_salt_1m_from_surf_p1 = np.empty(len(turbidity_data_2020.n))

# Set the depth from the surface 
surf_depth = 1

# Loop through the points and interpolate onto these depths 
for k in range(len(turbidity_data_2020.n)):
    # Pull out the bottom depth
    bot_depth = obs_bottom_depths[k]
    # Interpolate onto depths
    obs_ssc_1m_above_seafloor_p1[k] = turbidity_data_2020.turbidity[:,k].interp(depth_intervals=bot_depth)
    obs_ssc_1m_from_surf_p1[k] = turbidity_data_2020.turbidity[:,k].interp(depth_intervals=surf_depth)
    obs_temp_1m_above_seafloor_p1[k] = turbidity_data_2020.temperature[:,k].interp(depth_intervals=bot_depth)
    obs_temp_1m_from_surf_p1[k] = turbidity_data_2020.temperature[:,k].interp(depth_intervals=surf_depth)
    obs_salt_1m_above_seafloor_p1[k] = turbidity_data_2020.salinity[:,k].interp(depth_intervals=bot_depth)
    obs_salt_1m_from_surf_p1[k] = turbidity_data_2020.salinity[:,k].interp(depth_intervals=surf_depth)
    


# ------------------------ Prep for Scatter Plot --------------------------------
# In order to make a scatter plot comparing the values in ROMS and the observations,
# need to interpolate the ROMS data onto the observation locations so they are
# the same size
# Before we can look at ROMS SSC compared to Eidam, we need
# to interpolate ssc from ROMS onto Eidam points

# Set the input and output grids, and sepcify the lat/lon
# Since we are looking at ssc, we will use lon_rho and lat_rho as the primary lat/lon for the grid 
# ROMS rho input grid 
ds_in_rho = grid.copy() # need to use lon_180 for this grid 
ds_in_rho['lon'] = (('eta_rho', 'xi_rho'), ds_in_rho.lon_rho.values)
ds_in_rho['lat'] = (('eta_rho', 'xi_rho'), ds_in_rho.lat_rho.values)

# Output grid (ROMS rho grid)
turbidity_data_2020.lon[:36].values
ds_out_eidam = turbidity_data_2020.copy()
ds_out_eidam['lat'] = (('n'), ds_out_eidam.lat.values)
ds_out_eidam['lon'] = (('n'), ds_out_eidam.lon.values)

# Add masks 
# ex: ds["mask"] = xr.where(~np.isnan(ds["zeta"].isel(ocean_time=0)), 1, 0)
# Input grid (HYCOM)
# this is only a surface mask - which is what we want 
# ROMS rho grid
ds_in_rho['mask'] = (('eta_rho', 'xi_rho'), ds_in_rho.mask_rho.values)
# Output grid (Eidam data which does not have mask...)
#ds_out_rho['mask'] = (('eta_rho', 'xi_rho'), ds_out_rho.mask_rho.values)

# Regrid from u grid to rho grid with the masks included and extrapolation used 
regridder_rho2eidam = xe.Regridder(ds_in_rho, ds_out_eidam, method="bilinear", extrap_method='inverse_dist') #extrap_method="nearest_s2d"
regridder_rho2eidam

# Try regridding the ROMS data?
# Surface
roms_ssc_1m_from_surf_on_eidam_points_tmp = regridder_rho2eidam(roms_ssc_1m_from_surf_p1_avg)
roms_temp_1m_from_surf_on_eidam_points_tmp = regridder_rho2eidam(roms_temp_1m_from_surf_p1_avg)
roms_salt_1m_from_surf_on_eidam_points_tmp = regridder_rho2eidam(roms_salt_1m_from_surf_p1_avg)
# Bottom 
roms_ssc_1m_above_seafloor_on_eidam_points_tmp = regridder_rho2eidam(roms_ssc_1m_above_seafloor_p1_avg)
roms_temp_1m_above_seafloor_on_eidam_points_tmp = regridder_rho2eidam(roms_temp_1m_above_seafloor_p1_avg)
roms_salt_1m_above_seafloor_on_eidam_points_tmp = regridder_rho2eidam(roms_salt_1m_above_seafloor_p1_avg)

# Resize to take just those points (the middle)
# Make empty arrays to hold values
# Surface
roms_ssc_1m_from_surf_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))
roms_temp_1m_from_surf_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))
roms_salt_1m_from_surf_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))
# Bottom 
roms_ssc_1m_above_seafloor_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))
roms_temp_1m_above_seafloor_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))
roms_salt_1m_above_seafloor_on_eidam_points = np.empty((len(turbidity_data_2020.lon)))

# Loop through points 
for g in range(len(turbidity_data_2020.lon)):
    # Pull out just the values we want and save it
    # Surface
    roms_ssc_1m_from_surf_on_eidam_points[g] = roms_ssc_1m_from_surf_on_eidam_points_tmp[g,g]
    roms_temp_1m_from_surf_on_eidam_points[g] = roms_temp_1m_from_surf_on_eidam_points_tmp[g,g]
    roms_salt_1m_from_surf_on_eidam_points[g] = roms_salt_1m_from_surf_on_eidam_points_tmp[g,g]
    # Bottom 
    roms_ssc_1m_above_seafloor_on_eidam_points[g] = roms_ssc_1m_above_seafloor_on_eidam_points_tmp[g,g]
    roms_temp_1m_above_seafloor_on_eidam_points[g] = roms_temp_1m_above_seafloor_on_eidam_points_tmp[g,g]
    roms_salt_1m_above_seafloor_on_eidam_points[g] = roms_salt_1m_above_seafloor_on_eidam_points_tmp[g,g]








# --------------------------------------------------------------------------------
# --------- Plot 1: Subplot of SSC 1m Above Seafloor and 1m Below Surface --------
# -------------------------- Obs and ROMS ----------------------------------------
# --------------------------------------------------------------------------------
# Make subplots of the different levels of tempertaure, salinity, and SSC
# with ROMS in contours and observations in scatter plot on top

# Mask the data 
# Set the number of cells in the sponge on each open boundary
c_west = 36
c_north = 45
c_east = 36

# Make land gray
# Make mask
temp_mask = grid.mask_rho.values
temp_mask = np.where(temp_mask==0, np.nan, temp_mask)
# Replace nans so it will be gray
#tot_mud_percent_wland = tot_mud_percent * temp_mask
#tot_mud_percent_wland = tot_mud_percent_wland.fillna(-1)
#bstrcwmax_wland = np.nan_to_num(bstrcwmax_wland, nan=-100)
# ROMS SSC
roms_ssc_1m_above_seafloor_p1_avg_wland = roms_ssc_1m_above_seafloor_p1_avg*temp_mask
roms_ssc_1m_above_seafloor_p1_avg_wland = roms_ssc_1m_above_seafloor_p1_avg_wland.fillna(-100)
roms_ssc_1m_from_surf_p1_avg_wland = roms_ssc_1m_from_surf_p1_avg*temp_mask
roms_ssc_1m_from_surf_p1_avg_wland = roms_ssc_1m_from_surf_p1_avg_wland.fillna(-100)
# ROMS temp
roms_temp_1m_above_seafloor_p1_avg_wland = roms_temp_1m_above_seafloor_p1_avg*temp_mask
roms_temp_1m_above_seafloor_p1_avg_wland = np.nan_to_num(roms_temp_1m_above_seafloor_p1_avg_wland, nan=-100)
roms_temp_1m_from_surf_p1_avg_wland = roms_temp_1m_from_surf_p1_avg*temp_mask
roms_temp_1m_from_surf_p1_avg_wland = np.nan_to_num(roms_temp_1m_from_surf_p1_avg_wland, nan=-100)
# ROMS salt 
roms_salt_1m_above_seafloor_p1_avg_wland = roms_salt_1m_above_seafloor_p1_avg*temp_mask
roms_salt_1m_above_seafloor_p1_avg_wland = roms_salt_1m_above_seafloor_p1_avg_wland.fillna(-100)
roms_salt_1m_from_surf_p1_avg_wland = roms_salt_1m_from_surf_p1_avg*temp_mask
roms_salt_1m_from_surf_p1_avg_wland = roms_salt_1m_from_surf_p1_avg_wland.fillna(-100)



# Prep the data by  multiplying by the mask and trimming

# Multiply by mask
h_masked = grid.h.values*grid.mask_rho.values*mask_rho_nan.nudge_mask_rho_nan
# ROMS 
roms_ssc_1m_above_seafloor_p1_avg_wland_masked = roms_ssc_1m_above_seafloor_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan
roms_ssc_1m_from_surf_p1_avg_wland_masked = roms_ssc_1m_from_surf_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan
roms_temp_1m_above_seafloor_p1_avg_wland_masked = roms_temp_1m_above_seafloor_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan
roms_temp_1m_from_surf_p1_avg_wland_masked = roms_temp_1m_from_surf_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan
roms_salt_1m_above_seafloor_p1_avg_wland_masked = roms_salt_1m_above_seafloor_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan
roms_salt_1m_from_surf_p1_avg_wland_masked = roms_salt_1m_from_surf_p1_avg_wland*mask_rho_nan.nudge_mask_rho_nan


# Trim 
lon_rho_trimmed = grid.lon_rho[:,c_west:-c_west].values
lat_rho_trimmed = grid.lat_rho[:,c_west:-c_west].values
h_masked_trimmed = h_masked[:,c_west:-c_west]
# ROMS 
roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim = roms_ssc_1m_above_seafloor_p1_avg_wland_masked[:,c_west:-c_west]
roms_ssc_1m_from_surf_p1_avg_wland_masked_trim = roms_ssc_1m_from_surf_p1_avg_wland_masked[:,c_west:-c_west]
roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim = roms_temp_1m_above_seafloor_p1_avg_wland_masked[:,c_west:-c_west]
roms_temp_1m_from_surf_p1_avg_wland_masked_trim = roms_temp_1m_from_surf_p1_avg_wland_masked[:,c_west:-c_west]
roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim = roms_salt_1m_above_seafloor_p1_avg_wland_masked[:,c_west:-c_west]
roms_salt_1m_from_surf_p1_avg_wland_masked_trim = roms_salt_1m_from_surf_p1_avg_wland_masked[:,c_west:-c_west]


# Make reverse mask so land can be gray 
reverse_colors = colors.ListedColormap(['darkgray','#FF000000'])
bounds=[0,1,2]
norm = colors.BoundaryNorm(bounds, reverse_colors.N)

# Make new mask so land is gray - NEW
mask_rho2 = grid.mask_rho.values
mask_rho2[mask_rho2==1.0] = 2.0
mask_rho2_trimmed = mask_rho2[:,c_west:-c_west]


# Make the plot
fig1, ax1 = plt.subplots(3, 2, figsize=(22,17))

# Set the color levels 
lev1 = [10,20,30,40,50,60,70,80,90,100]
#lev2 = np.arange(-0.5, 8, 0.1) # temp
lev2 = np.arange(-3, 6, 0.1) # temp
lev3 = np.arange(20, 32, 0.1) # salt
#lev4 = np.arange(0, 25, 0.1) # ssc
lev4 = np.arange(0, 20, 0.1) # ssc

# Set the colormaps 
# Temp
cmap2 = cmocean.cm.amp
#cmap2.set_under('darkgray')
# Salt 
cmap3 = cmocean.cm.haline
#cmap3.set_under('darkgray')
# SSC
cmap4 = cmocean.cm.turbid
#cmap4.set_under('darkgray')


# Surface temperature 
# Plot ROMS
cs1 = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc1 = ax1[0,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_from_surf_p1), vmin=-0.5, 
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0a = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs2 = ax1[0,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[0,0].get_xticklabels(), visible=False)
plt.setp(ax1[0,0].get_yticklabels(), visible=True)
ax1[0,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax1[0,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')

# Bottom temperature
# Plot ROMS
cs3 = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc2 = ax1[0,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1), vmin=-0.5,
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0b = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs4 = ax1[0,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax1[0,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')
plt.setp(ax1[0,1].get_xticklabels(), visible=False)
plt.setp(ax1[0,1].get_yticklabels(), visible=False)

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar2_ax = fig1.add_axes([0.81, 0.69, 0.015, 0.27])
#cbar2 = plt.colorbar(cs3, ax=[ax1[0,0], ax1[0,1]], cax=cbar2_ax, orientation='vertical').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize)
# If horizontal and in axes
cbar2 = plt.colorbar(cs3, cax=ax1[0,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[-3, -1, 1, 3, 5], ax=[ax1[0,0], ax1[0,1]], 
                     orientation='horizontal').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize-2)

# Surface salt
# Plot ROMS
cs5 = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
#All points
#sc3 = ax1[1,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_from_surf_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0c = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs6 = ax1[1,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[1,0].get_xticklabels(), visible=False)
ax1[1,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax1[1,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')


# Bottom salt
# Plot ROMS
cs7 = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
# All points 
#sc4 = ax1[1,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0d = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs8 = ax1[1,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[1,1].get_xticklabels(), visible=False)
plt.setp(ax1[1,1].get_yticklabels(), visible=False)
ax1[1,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar3_ax = fig1.add_axes([0.81, 0.395, 0.015, 0.27])
#cbar3 = plt.colorbar(cs7, ax=[ax1[1,0], ax1[1,1]], cax=cbar3_ax, orientation='vertical').set_label(label='Salinity (PSU)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs7, cax=ax1[1,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[20, 23, 26, 29, 32], ax=[ax1[1,0], ax1[1,1]], 
                     orientation='horizontal').set_label(label='Salinity (PSU)', size=fontsize-2)


# Surface SSC
# Plot ROMS
cs9 = ax1[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_ssc_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='both')
# Plot obs
# All points
#sc5 = ax1[2,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_from_surf_p1*0.8),  vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0e = ax1[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs10 = ax1[2,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax1[2,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax1[2,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')


# Bottom SSC
# Plot ROMS
cs11 = ax1[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                         roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='both')
# Plot obs
# All points
#sc6 = ax1[2,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1*0.8), vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0f = ax1[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs12 = ax1[2,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[2,1].get_yticklabels(), visible=False)
ax1[2,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar4_ax = fig1.add_axes([0.81, 0.1, 0.015, 0.27])
#cbar4 = plt.colorbar(cs9, ax=[ax1[2,0], ax1[2,1]], cax=cbar4_ax, orientation='vertical').set_label(label='SSC (mg/L)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs9, cax=ax1[2,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[0, 5, 10, 15, 20], ax=[ax1[2,0], ax1[2,1]], 
                     orientation='horizontal').set_label(label='SSC (mg/L)', size=fontsize-2)



# Adjust spacing
fig1.subplots_adjust(hspace=0.1, wspace=0.06)
# Common xy label
# If vertical outside
#fig1.text(0.04, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
#fig1.text(0.5, 0.04, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# If horizontal inside
fig1.text(0.06, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
fig1.text(0.5, 0.06, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# Add subplot labels
# If vertical outside axes
#plt.text(0.105, 0.694, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.694, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.401, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.401, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.110, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.110, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
# If horizontal inside axes
plt.text(0.132, 0.654, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.530, 0.654, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.132, 0.397, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.529, 0.397, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.132, 0.135, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.531, 0.135, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)





# --------------------------------------------------------------------------------
# --------- Plot 2: Same as above but different format -------------------------------
# --------------------------------------------------------------------------------
# Same as above but slightly different formatting 

# Make the plot
fig1, ax1 = plt.subplots(3, 2, figsize=(22,17))

# Set the color levels 
lev1 = [10,20,30,40,50,60,70,80,90,100]
#lev2 = np.arange(-0.5, 8, 0.1) # temp
lev2 = np.arange(-3, 6, 0.1) # temp
lev3 = np.arange(20, 32, 0.1) # salt
#lev4 = np.arange(0, 25, 0.1) # ssc
lev4 = np.arange(0, 20, 0.1) # ssc

# Set the colormaps 
# Temp
cmap2 = cmocean.cm.amp
#cmap2.set_under('darkgray')
# Salt 
cmap3 = cmocean.cm.haline
#cmap3.set_under('darkgray')
# SSC
cmap4 = cmocean.cm.turbid
#cmap4.set_under('darkgray')


# Surface temperature 
# Plot ROMS
cs1 = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc1 = ax1[0,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_from_surf_p1), vmin=-0.5, 
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0a = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs2 = ax1[0,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[0,0].get_xticklabels(), visible=False)
plt.setp(ax1[0,0].get_yticklabels(), visible=True)
ax1[0,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax1[0,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax1[0,0].set_ylabel('Temperature', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=70, va='center')

# Bottom temperature
# Plot ROMS
cs3 = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc2 = ax1[0,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1), vmin=-0.5,
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0b = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs4 = ax1[0,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax1[0,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')
plt.setp(ax1[0,1].get_xticklabels(), visible=False)
plt.setp(ax1[0,1].get_yticklabels(), visible=False)

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar2_ax = fig1.add_axes([0.81, 0.69, 0.015, 0.27])
#cbar2 = plt.colorbar(cs3, ax=[ax1[0,0], ax1[0,1]], cax=cbar2_ax, orientation='vertical').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize)
# If horizontal and in axes
cbar2 = plt.colorbar(cs3, cax=ax1[0,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[-3, -1, 1, 3, 5], ax=[ax1[0,0], ax1[0,1]], 
                     orientation='horizontal').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize-2)

# Surface salt
# Plot ROMS
cs5 = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
#All points
#sc3 = ax1[1,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_from_surf_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0c = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs6 = ax1[1,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[1,0].get_xticklabels(), visible=False)
ax1[1,0].yaxis.set_major_locator(plt.MaxNLocator(5))
#ax1[1,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax1[1,0].set_ylabel('Salinity', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=80, va='center')


# Bottom salt
# Plot ROMS
cs7 = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
# All points 
#sc4 = ax1[1,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0d = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs8 = ax1[1,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[1,1].get_xticklabels(), visible=False)
plt.setp(ax1[1,1].get_yticklabels(), visible=False)
#ax1[1,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar3_ax = fig1.add_axes([0.81, 0.395, 0.015, 0.27])
#cbar3 = plt.colorbar(cs7, ax=[ax1[1,0], ax1[1,1]], cax=cbar3_ax, orientation='vertical').set_label(label='Salinity (PSU)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs7, cax=ax1[1,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[20, 23, 26, 29, 32], ax=[ax1[1,0], ax1[1,1]], 
                     orientation='horizontal').set_label(label='Salinity (PSU)', size=fontsize-2)


# Surface SSC
# Plot ROMS
cs9 = ax1[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_ssc_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='max')
# Plot obs
# All points
#sc5 = ax1[2,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_from_surf_p1*0.8),  vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc5 = ax1[2,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0e = ax1[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs10 = ax1[2,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax1[2,0].yaxis.set_major_locator(plt.MaxNLocator(5))
#ax1[2,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax1[2,0].set_ylabel('SSC', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=70, va='center')


# Bottom SSC
# Plot ROMS
cs11 = ax1[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                         roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='max')
# Plot obs
# All points
#sc6 = ax1[2,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1*0.8), vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc6 = ax1[2,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0f = ax1[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs12 = ax1[2,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[2,1].get_yticklabels(), visible=False)
#ax1[2,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar4_ax = fig1.add_axes([0.81, 0.1, 0.015, 0.27])
#cbar4 = plt.colorbar(cs9, ax=[ax1[2,0], ax1[2,1]], cax=cbar4_ax, orientation='vertical').set_label(label='SSC (mg/L)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs9, cax=ax1[2,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[0, 5, 10, 15, 20], ax=[ax1[2,0], ax1[2,1]], 
                     orientation='horizontal').set_label(label='SSC (mg/L)', size=fontsize-2)



# Adjust spacing
fig1.subplots_adjust(hspace=0.1, wspace=0.06)
# Common xy label
# If vertical outside
#fig1.text(0.04, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
#fig1.text(0.5, 0.04, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# If horizontal inside
fig1.text(0.075, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize) # 0.06, 0.5
fig1.text(0.5, 0.06, 'Longitude (degrees)', ha='center', fontsize=fontsize)
# Add subplot labels
# If vertical outside axes
#plt.text(0.105, 0.694, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.694, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.401, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.401, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.110, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.110, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
# If horizontal inside axes
plt.text(0.132, 0.654, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.530, 0.654, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.132, 0.397, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.529, 0.397, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.132, 0.135, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.531, 0.135, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)






# --------------------------------------------------------------------------------
# -- Plot 2b: Same as above but different format with Scatter Plot Added In ------
# --------------------------------------------------------------------------------
# Same as above but slightly different formatting 

# Make the plot
fig2, ax2 = plt.subplots(3, 3, figsize=(28,17), gridspec_kw={'width_ratios': [0.7, 0.7, 0.5]}) # (22,17)

# Set the color levels 
lev1 = [10,20,30,40,50,60,70,80,90,100]
#lev2 = np.arange(-0.5, 8, 0.1) # temp
lev2 = np.arange(-3, 6, 0.1) # temp
lev3 = np.arange(20, 32, 0.1) # salt
#lev4 = np.arange(0, 25, 0.1) # ssc
lev4 = np.arange(0, 20, 0.1) # ssc


# Surface temperature 
# Plot ROMS
cs13 = ax2[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc1 = ax1[0,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_from_surf_p1), vmin=-0.5, 
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc7 = ax2[0,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc7 = ax2[0,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc7 = ax2[0,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0a = ax2[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs14 = ax2[0,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax2[0,0].get_xticklabels(), visible=False)
plt.setp(ax2[0,0].get_yticklabels(), visible=True)
ax2[0,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax2[0,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax2[0,0].set_ylabel('Temperature', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=70, va='center')

# Bottom temperature
# Plot ROMS
cs15 = ax2[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both')
# Plot obs
# All points
#sc2 = ax1[0,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1), vmin=-0.5,
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc8 = ax2[0,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                      marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc8 = ax2[0,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc8 = ax2[0,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0b = ax2[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs16 = ax2[0,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax2[0,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')
plt.setp(ax2[0,1].get_xticklabels(), visible=False)
plt.setp(ax2[0,1].get_yticklabels(), visible=False)

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar2_ax = fig1.add_axes([0.81, 0.69, 0.015, 0.27])
#cbar2 = plt.colorbar(cs3, ax=[ax1[0,0], ax1[0,1]], cax=cbar2_ax, orientation='vertical').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize)
# If horizontal and in axes
cbar2 = plt.colorbar(cs15, cax=ax2[0,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[-3, -1, 1, 3, 5], ax=[ax1[0,0], ax1[0,1]], 
                     orientation='horizontal').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize-2)

# Surface salt
# Plot ROMS
cs17 = ax2[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
#All points
#sc3 = ax1[1,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_from_surf_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc9 = ax2[1,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc9 = ax2[1,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc9 = ax2[1,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0c = ax2[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs18 = ax2[1,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax2[1,0].get_xticklabels(), visible=False)
ax2[1,0].yaxis.set_major_locator(plt.MaxNLocator(5))
#ax1[1,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax2[1,0].set_ylabel('Salinity', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=80, va='center')


# Bottom salt
# Plot ROMS
cs19 = ax2[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Plot obs
# All points 
#sc4 = ax1[1,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc10 = ax2[1,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc10 = ax2[1,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc10 = ax2[1,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0d = ax2[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs20 = ax2[1,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax2[1,1].get_xticklabels(), visible=False)
plt.setp(ax2[1,1].get_yticklabels(), visible=False)
#ax1[1,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar3_ax = fig1.add_axes([0.81, 0.395, 0.015, 0.27])
#cbar3 = plt.colorbar(cs7, ax=[ax1[1,0], ax1[1,1]], cax=cbar3_ax, orientation='vertical').set_label(label='Salinity (PSU)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs19, cax=ax2[1,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[20, 23, 26, 29, 32], ax=[ax1[1,0], ax1[1,1]], 
                     orientation='horizontal').set_label(label='Salinity (PSU)', size=fontsize-2)


# Surface SSC
# Plot ROMS
cs21 = ax2[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_ssc_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='both')
# Plot obs
# All points
#sc5 = ax1[2,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_from_surf_p1*0.8),  vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc11 = ax2[2,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc11 = ax2[2,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc11 = ax2[2,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0e = ax2[2,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs22 = ax2[2,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax2[2,0].yaxis.set_major_locator(plt.MaxNLocator(5))
#ax1[2,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax2[2,0].set_ylabel('SSC', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=70, va='center')


# Bottom SSC
# Plot ROMS
cs23 = ax2[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                         roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='max')
# Plot obs
# All points
#sc6 = ax1[2,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1*0.8), vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc12 = ax2[2,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc12 = ax2[2,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc12 = ax2[2,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0f = ax2[2,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs24 = ax2[2,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax2[2,1].get_yticklabels(), visible=False)
#ax1[2,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar4_ax = fig1.add_axes([0.81, 0.1, 0.015, 0.27])
#cbar4 = plt.colorbar(cs9, ax=[ax1[2,0], ax1[2,1]], cax=cbar4_ax, orientation='vertical').set_label(label='SSC (mg/L)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs23, cax=ax2[2,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[0, 5, 10, 15, 20], ax=[ax1[2,0], ax1[2,1]], 
                     orientation='horizontal').set_label(label='SSC (mg/L)', size=fontsize-2)

# Add in a whole section of scatter plots 
# SSC
# Surface
ax2[0,2].scatter(roms_temp_1m_from_surf_on_eidam_points[:36], obs_temp_1m_above_seafloor_p1[:36], 
            marker='.', s=80, color='royalblue', label='1m From Surface')
ax2[0,2].scatter(roms_temp_1m_from_surf_on_eidam_points[37:60], obs_temp_1m_above_seafloor_p1[37:60], 
            marker='.', s=80, color='royalblue')
ax2[0,2].scatter(roms_temp_1m_from_surf_on_eidam_points[61:90], obs_temp_1m_above_seafloor_p1[61:90], 
            marker='.', s=80, color='royalblue')
# Bottom 
ax2[0,2].scatter(roms_temp_1m_above_seafloor_on_eidam_points[:36], obs_temp_1m_above_seafloor_p1[:36], 
            marker='.', s=80,  color='red', label='1m Above Seafloor')
ax2[0,2].scatter(roms_temp_1m_above_seafloor_on_eidam_points[37:60], obs_temp_1m_above_seafloor_p1[37:60], 
            marker='.', s=80, color='red')
ax2[0,2].scatter(roms_temp_1m_above_seafloor_on_eidam_points[61:90], obs_temp_1m_above_seafloor_p1[61:90], 
            marker='.', s=80, color='red')
# Add 1:1 lines
# Both (they are the same)
min_val_temp_seafloor = min(min(roms_temp_1m_above_seafloor_on_eidam_points), min(obs_temp_1m_above_seafloor_p1))
max_val_temp_seafloor = max(max(roms_temp_1m_above_seafloor_on_eidam_points), max(obs_temp_1m_above_seafloor_p1))
ax2[0,2].plot([min_val_temp_seafloor, max_val_temp_seafloor], [min_val_temp_seafloor, max_val_temp_seafloor], 'k', label='1:1 Line')
# Add a legend
#ax2[0,2].legend(fontsize=fontsize-2)
# Label the axes
ax2[0,2].set_xlabel('Modeled Temperature (\N{DEGREE SIGN}C)', fontsize=fontsize-2)
ax2[0,2].set_ylabel('Observed Temperature (\N{DEGREE SIGN}C)', fontsize=fontsize-2)

# Salinity
# Surface
ax2[1,2].scatter(roms_salt_1m_from_surf_on_eidam_points[:36], obs_salt_1m_from_surf_p1[:36], 
            marker='.', s=80, color='royalblue', label='1m From Surface')
ax2[1,2].scatter(roms_salt_1m_from_surf_on_eidam_points[37:60], obs_salt_1m_from_surf_p1[37:60], 
            marker='.', s=80, color='royalblue')
ax2[1,2].scatter(roms_salt_1m_from_surf_on_eidam_points[61:90], obs_salt_1m_from_surf_p1[61:90], 
            marker='.', s=80, color='royalblue')
# Bottom 
ax2[1,2].scatter(roms_salt_1m_above_seafloor_on_eidam_points[:36], obs_salt_1m_above_seafloor_p1[:36], 
            marker='.', s=80, color='red', label='1m Above Seafloor')
ax2[1,2].scatter(roms_salt_1m_above_seafloor_on_eidam_points[37:60], obs_salt_1m_above_seafloor_p1[37:60], 
            marker='.', s=80, color='red')
ax2[1,2].scatter(roms_salt_1m_above_seafloor_on_eidam_points[61:90], obs_salt_1m_above_seafloor_p1[61:90], 
            marker='.', s=80, color='red')
# Add 1:1 lines
# Both (they are the same)
min_val_salt_seafloor = min(min(roms_salt_1m_above_seafloor_on_eidam_points), min(obs_salt_1m_above_seafloor_p1))
max_val_salt_seafloor = max(max(roms_salt_1m_above_seafloor_on_eidam_points), max(obs_salt_1m_above_seafloor_p1))
ax2[1,2].plot([min_val_salt_seafloor, max_val_salt_seafloor], [min_val_salt_seafloor, max_val_salt_seafloor], 'k', label='1:1 Line')
# Add a legend
ax2[1,2].legend(fontsize=fontsize-2, loc='upper right')
# Label the axes
ax2[1,2].set_xlabel('Modeled Salinity (PSU)', fontsize=fontsize-2)
ax2[1,2].set_ylabel('Observed Salinity (PSU)', fontsize=fontsize-2)

# SSC
# Surface
ax2[2,2].scatter(roms_ssc_1m_from_surf_on_eidam_points[:36]*1000, obs_ssc_1m_from_surf_p1[:36], 
            marker='.', s=80, color='royalblue', label='1m From Surface')
ax2[2,2].scatter(roms_ssc_1m_from_surf_on_eidam_points[37:60]*1000, obs_ssc_1m_from_surf_p1[37:60], 
            marker='.', s=80, color='royalblue')
ax2[2,2].scatter(roms_ssc_1m_from_surf_on_eidam_points[61:90]*1000, obs_ssc_1m_from_surf_p1[61:90], 
            marker='.', s=80, color='royalblue')
# Bottom 
ax2[2,2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[:36]*1000, obs_ssc_1m_above_seafloor_p1[:36], 
            marker='.', s=80, color='red', label='1m Above Seafloor')
ax2[2,2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[37:60]*1000, obs_ssc_1m_above_seafloor_p1[37:60], 
            marker='.', s=80, color='red')
ax2[2,2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[61:90]*1000, obs_ssc_1m_above_seafloor_p1[61:90], 
            marker='.', s=80, color='red')
# Add 1:1 lines
# Both (they are the same)
min_val_seafloor = min(min(roms_ssc_1m_above_seafloor_on_eidam_points), min(obs_ssc_1m_above_seafloor_p1))
max_val_seafloor = max(max(roms_ssc_1m_above_seafloor_on_eidam_points), max(obs_ssc_1m_above_seafloor_p1))
ax2[2,2].plot([min_val_seafloor, max_val_seafloor], [min_val_seafloor, max_val_seafloor], 'k', label='1:1 Line')
# Add a legend
#ax2[2,2].legend(fontsize=fontsize-2)
# Label the axes
ax2[2,2].set_xlabel('Modeled SSC (mg/L)', fontsize=fontsize-2)
ax2[2,2].set_ylabel('Observed SSC (mg/L)', fontsize=fontsize-2)

# Adjust spacing
fig2.subplots_adjust(hspace=0.22, wspace=0.17)
# Common xy label
# If vertical outside
#fig1.text(0.04, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
#fig1.text(0.5, 0.04, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# If horizontal inside
fig2.text(0.081, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize) # 0.06, 0.5
fig2.text(0.4, 0.07, 'Longitude (degrees)', ha='center', fontsize=fontsize)
# Add subplot labels
# If vertical outside axes
#plt.text(0.105, 0.694, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.694, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.401, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.401, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.110, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.110, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
# If horizontal inside axes
plt.text(0.131, 0.669, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.424, 0.669, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.887, 0.669, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.131, 0.403, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.426, 0.403, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.887, 0.403, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.131, 0.136, 'g)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.430, 0.136, 'h)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.887, 0.136, 'i)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)





# Make the same plot as above but split into one for just SSC with the scatter
# plot and then another with just the maps for salt and temp
# --------------------------------------------------------------------------------
# -- Plot 2c: Same as above but only SSC ------
# --------------------------------------------------------------------------------
# Same as above but with only SSC data (maps and scatter plot)

# Make the plot
fig2, ax2 = plt.subplots(1, 3, figsize=(32,6), gridspec_kw={'width_ratios': [0.7, 0.7, 0.5]}) # (22,17)

# Surface SSC
# Plot ROMS
cs21 = ax2[0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_ssc_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='both', projection=crs)
# Plot obs
# All points
#sc5 = ax1[2,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_from_surf_p1*0.8),  vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc11 = ax2[0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc11 = ax2[0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc11 = ax2[0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_from_surf_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0e = ax2[0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs22 = ax2[0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax2[0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax2[0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
#ax1[2,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
#ax2[0].set_ylabel('SSC', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=70, va='center')


# Bottom SSC
# Plot ROMS
cs23 = ax2[1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                         roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140]*1000, 
                        lev4, cmap=cmap4, extend='max', projection=crs)
# Plot obs
# All points
#sc6 = ax1[2,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1*0.8), vmin=0,
  #                     vmax=25, edgecolor='black', linewidth=3, cmap=cmap4)
# Non-shelf break points (36, 60, 90, 91)
sc12 = ax2[1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[:36]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc12 = ax2[1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[37:60]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
sc12 = ax2[1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_ssc_1m_above_seafloor_p1[61:90]), vmin=0, 
                       vmax=20, edgecolor='black', linewidth=3, cmap=cmap4)
# Plot land as gray 
cs0f = ax2[1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs24 = ax2[1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax2[1].get_yticklabels(), visible=False)
ax2[1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')
#ax1[2,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar4_ax = fig1.add_axes([0.81, 0.1, 0.015, 0.27])
#cbar4 = plt.colorbar(cs9, ax=[ax1[2,0], ax1[2,1]], cax=cbar4_ax, orientation='vertical').set_label(label='SSC (mg/L)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs23, cax=ax2[1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[0, 5, 10, 15, 20], ax=[ax2[0], ax2[1]], 
                     orientation='horizontal').set_label(label='SSC (mg/L)', size=fontsize-2)

# Add in a whole section of scatter plots 
# SSC
# Surface
ax2[2].scatter(roms_ssc_1m_from_surf_on_eidam_points[:36]*1000, obs_ssc_1m_from_surf_p1[:36], 
            marker='.', s=80, color='royalblue', label='1m From Surface')
ax2[2].scatter(roms_ssc_1m_from_surf_on_eidam_points[37:60]*1000, obs_ssc_1m_from_surf_p1[37:60], 
            marker='.', s=80, color='royalblue')
ax2[2].scatter(roms_ssc_1m_from_surf_on_eidam_points[61:90]*1000, obs_ssc_1m_from_surf_p1[61:90], 
            marker='.', s=80, color='royalblue')
# Bottom 
ax2[2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[:36]*1000, obs_ssc_1m_above_seafloor_p1[:36], 
            marker='.', s=80, color='red', label='1m Above Seafloor')
ax2[2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[37:60]*1000, obs_ssc_1m_above_seafloor_p1[37:60], 
            marker='.', s=80, color='red')
ax2[2].scatter(roms_ssc_1m_above_seafloor_on_eidam_points[61:90]*1000, obs_ssc_1m_above_seafloor_p1[61:90], 
            marker='.', s=80, color='red')
# Add 1:1 lines
# Both (they are the same)
min_val_seafloor = min(min(roms_ssc_1m_above_seafloor_on_eidam_points), min(obs_ssc_1m_above_seafloor_p1))
max_val_seafloor = max(max(roms_ssc_1m_above_seafloor_on_eidam_points), max(obs_ssc_1m_above_seafloor_p1))
ax2[2].plot([min_val_seafloor, max_val_seafloor], [min_val_seafloor, max_val_seafloor], 'k', label='1:1 Line')
# Add a legend
ax2[2].legend(fontsize=fontsize-2)
# Label the axes
ax2[2].set_xlabel('Modeled SSC (mg/L)', fontsize=fontsize-2)
ax2[2].set_ylabel('Observed SSC (mg/L)', fontsize=fontsize-2)

# Adjust spacing
fig2.subplots_adjust(hspace=0.22, wspace=0.17)
# Common xy label
# If vertical outside
#fig1.text(0.04, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
#fig1.text(0.5, 0.04, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# If horizontal inside
fig2.text(0.081, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=fontsize) # 0.06, 0.5
fig2.text(0.4, -0.03, 'Longitude', ha='center', fontsize=fontsize)
# Add subplot labels
# If horizontal inside axes
plt.text(0.129, 0.139, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.428, 0.139, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.887, 0.139, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)




# --------------------------------------------------------------------------------
# ------ Plot 2d: Maps for just Salt & Temp with not scatter plots ---------------
# --------------------------------------------------------------------------------
# Same as above but slightly different formatting 

# Make the plot
fig1, ax1 = plt.subplots(2, 2, figsize=(24,12))

# Surface temperature 
# Plot ROMS
cs1 = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both', projection=crs)
# Plot obs
# All points
#sc1 = ax1[0,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_from_surf_p1), vmin=-0.5, 
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc1 = ax1[0,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_from_surf_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0a = ax1[0,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs2 = ax1[0,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax1[0,0].get_xticklabels(), visible=False)
plt.setp(ax1[0,0].get_yticklabels(), visible=True)
ax1[0,0].yaxis.set_major_locator(plt.MaxNLocator(5))
ax1[0,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax1[0,0].set_ylabel('Temperature', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=80, va='center')

# Bottom temperature
# Plot ROMS
cs3 = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev2, cmap=cmap2, extend='both', projection=crs)
# Plot obs
# All points
#sc2 = ax1[0,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1), vmin=-0.5,
  #                     vmax=8, edgecolor='black', linewidth=3, cmap=cmap2)
# Non-shelf break points (36, 60, 90, 91)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[:36]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[37:60]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
sc2 = ax1[0,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_temp_1m_above_seafloor_p1[61:90]), vmin=-3, 
                       vmax=6, edgecolor='black', linewidth=3, cmap=cmap2)
# Plot land as gray 
cs0b = ax1[0,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs4 = ax1[0,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
ax1[0,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')
plt.setp(ax1[0,1].get_xticklabels(), visible=False)
plt.setp(ax1[0,1].get_yticklabels(), visible=False)

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar2_ax = fig1.add_axes([0.81, 0.69, 0.015, 0.27])
#cbar2 = plt.colorbar(cs3, ax=[ax1[0,0], ax1[0,1]], cax=cbar2_ax, orientation='vertical').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize)
# If horizontal and in axes
cbar2 = plt.colorbar(cs3, cax=ax1[0,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[-3, -1, 1, 3, 5], ax=[ax1[0,0], ax1[0,1]], 
                     orientation='horizontal').set_label(label='Temperature (\N{DEGREE SIGN}C)', size=fontsize-2)

# Surface salt
# Plot ROMS
cs5 = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both', projection=crs)
# Plot obs
#All points
#sc3 = ax1[1,0].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_from_surf_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc3 = ax1[1,0].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0c = ax1[1,0].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs6 = ax1[1,0].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
#plt.setp(ax1[1,0].get_xticklabels(), visible=False)
ax1[1,0].yaxis.set_major_locator(plt.MaxNLocator(5))
#ax1[1,0].set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
ax1[1,0].set_ylabel('Salinity', fontsize=fontsize, fontweight='bold', rotation=0, labelpad=80, va='center')


# Bottom salt
# Plot ROMS
cs7 = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both', projection=crs)
# Plot obs
# All points 
#sc4 = ax1[1,1].scatter(turbidity_data_2020.lon.values, turbidity_data_2020.lat.values, 
 #                      marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1), vmin=20,
  #                     vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Non-shelf break points (36, 60, 90, 91)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[:36].values, turbidity_data_2020.lat[:36].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[:36]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[37:60].values, turbidity_data_2020.lat[37:60].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[37:60]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
sc4 = ax1[1,1].scatter(turbidity_data_2020.lon[61:90].values, turbidity_data_2020.lat[61:90].values, 
                       marker='.', s=800, c=(obs_salt_1m_above_seafloor_p1[61:90]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot land as gray 
cs0d = ax1[1,1].contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], mask_rho2_trimmed[:,50:-140], cmap=reverse_colors, norm=norm)
# Plot bathymetry contours 
cs8 = ax1[1,1].contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
#plt.setp(ax1[1,1].get_xticklabels(), visible=False)
plt.setp(ax1[1,1].get_yticklabels(), visible=False)
#ax1[1,1].set_title('1 m Above Seafloor', fontsize=fontsize+3, fontweight='bold')

# Set the colorbar for the temperature plots 
# Adjust colorbar placement
bottom, top = 0.1, 0.95
left, right = 0.1, 0.8
#fig1.subplots_adjust(top=top, bottom=bottom, left=left, right=right, wspace=0.1, hspace=0.2)
#axes = ax5.ravel().tolist()
#cbar3_ax = fig1.add_axes([0.81, 0.395, 0.015, 0.27])
#cbar3 = plt.colorbar(cs7, ax=[ax1[1,0], ax1[1,1]], cax=cbar3_ax, orientation='vertical').set_label(label='Salinity (PSU)', size=fontsize)
# If horizontal and in axes
cbar3 = plt.colorbar(cs7, cax=ax1[1,1].inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[20, 23, 26, 29, 32], ax=[ax1[1,0], ax1[1,1]], 
                     orientation='horizontal').set_label(label='Salinity (PSU)', size=fontsize-2)

# Adjust spacing
fig1.subplots_adjust(hspace=0.1, wspace=0.06)
# Common xy label
# If vertical outside
#fig1.text(0.04, 0.5, 'Latitude (degrees)', va='center', rotation='vertical', fontsize=fontsize+1)
#fig1.text(0.5, 0.04, 'Longitude (degrees)', ha='center', fontsize=fontsize+1)
# If horizontal inside
fig1.text(0.075, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=fontsize) # 0.06, 0.5
fig1.text(0.52, 0.06, 'Longitude', ha='center', fontsize=fontsize)
# Add subplot labels
# If vertical outside axes
#plt.text(0.105, 0.694, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.694, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.401, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.401, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.105, 0.110, 'e)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
#plt.text(0.475, 0.110, 'f)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
# If horizontal inside axes
plt.text(0.132, 0.535, 'a)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.530, 0.535, 'b)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.132, 0.135, 'c)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)
plt.text(0.531, 0.135, 'd)', fontsize=fontsize, fontweight='bold', color='k', transform=plt.gcf().transFigure)









# --------------------------------------------------------------------------------
# --------- Plot 3: Regions of Fresher Salt in Obs -------------------------------
# --------------------------------------------------------------------------------
# Plot the regions that are fresher in the observations so that we can pull out 
# some stats

fig3, ax3 = plt.subplots()
# ROMS
cs25 = ax3.contourf(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                        roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140], 
                        lev3, cmap=cmap3, extend='both')
# Obs
ax3.scatter(turbidity_data_2020.lon[0].values, turbidity_data_2020.lat[0].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[0]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
ax3.scatter(turbidity_data_2020.lon[18:22].values, turbidity_data_2020.lat[18:22].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[18:22]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
ax3.scatter(turbidity_data_2020.lon[25:29].values, turbidity_data_2020.lat[25:29].values, 
                       marker='.', s=800, c=(obs_salt_1m_from_surf_p1[25:29]), vmin=20, 
                       vmax=32, edgecolor='black', linewidth=3, cmap=cmap3)
# Plot bathymetry contours 
cs6 = ax3.contour(lon_rho_trimmed[:,50:-140], lat_rho_trimmed[:,50:-140], 
                       h_masked_trimmed[:,50:-140], lev1, colors='dimgray')
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
ax3.set_title('1 m Below Surface', fontsize=fontsize+3, fontweight='bold')
cbar33 = plt.colorbar(cs25, cax=ax3.inset_axes((0.38, 0.92, 0.6, 0.05)), 
                     ticks=[20, 23, 26, 29, 32], 
                     orientation='horizontal').set_label(label='Salinity (PSU)', size=fontsize-2)

# So those are the points we want - need to get the salinity in ROMS at these 
# points and subtract the salinity in the observations




##### Calculate the Mean Temperature Difference ######
# Caclculate the mean temperature difference between
# ROMS and the observations.

# Get the list of the lat/lon values 
# Get the lengths
# Longtidue
lon_len = len(turbidity_data_2020.lon[:36].values) + len(turbidity_data_2020.lon[37:60].values) + len(turbidity_data_2020.lon[61:90].values)
obs_lons = np.empty(lon_len)
# Latitude
lat_len = len(turbidity_data_2020.lat[:36].values) + len(turbidity_data_2020.lat[37:60].values) + len(turbidity_data_2020.lat[61:90].values)
obs_lats = np.empty(lat_len)

# Fill in the values 
# Longitude
obs_lons[:36] = turbidity_data_2020.lon[:36].values
obs_lons[36:59] = turbidity_data_2020.lon[37:60].values
obs_lons[59:] = turbidity_data_2020.lon[61:90].values
# Latitude
obs_lats[:36] = turbidity_data_2020.lat[:36].values
obs_lats[36:59] = turbidity_data_2020.lat[37:60].values
obs_lats[59:] = turbidity_data_2020.lat[61:90].values

# Find the ROMS temperatures at those same lat/lons
# Interpolate ROMS onto these locations (first interpolate to all locations, then
# select just the ones we want)
# Set input grid as ROMS rho
# Gridded
grid_in_rho = {"lon": grid.lon_rho.values, "lat": grid.lat_rho.values}
# Array
#grid_in_rho = {"lon": grid.lon_rho[0,:].values, "lat": grid.lat_rho[:,0].values}
# Set output grid as CTD 2020
# output grid has a larger coverage and finer resolution
grid_out_ctd_2020 = {"lon": turbidity_data_2020.lon.values, "lat": turbidity_data_2020.lat.values}

regridder_rho2ctd_2020 = xe.Regridder(grid_in_rho, grid_out_ctd_2020, "bilinear")
regridder_rho2ctd_2020

# Now interpolate 
# Temperature
# 1 m from surface
roms_temp_p1_eid_coords_1m_from_surf = regridder_rho2ctd_2020(roms_temp_1m_from_surf_p1_avg)
print(roms_temp_p1_eid_coords_1m_from_surf.shape)
# 1 m above seafloor
roms_temp_p1_eid_coords_1m_above_seabed = regridder_rho2ctd_2020(roms_temp_1m_above_seafloor_p1_avg)
print(roms_temp_p1_eid_coords_1m_above_seabed.shape)
# SSC
# 1 m from surface
roms_ssc_p1_eid_coords_1m_from_surf = regridder_rho2ctd_2020(roms_ssc_1m_from_surf_p1_avg)
print(roms_ssc_p1_eid_coords_1m_from_surf.shape)
# Convert to mg/L
roms_ssc_p1_eid_coords_1m_from_surf = roms_ssc_p1_eid_coords_1m_from_surf * 1000
# 1 m above seafloor
roms_ssc_p1_eid_coords_1m_above_seabed = regridder_rho2ctd_2020(roms_ssc_1m_above_seafloor_p1_avg)
print(roms_ssc_p1_eid_coords_1m_above_seabed.shape)
# Convert to mg/L
roms_ssc_p1_eid_coords_1m_above_seabed * 1000
# Salinity 
# 1 m from surface
roms_salt_p1_eid_coords_1m_from_surf = regridder_rho2ctd_2020(roms_salt_1m_from_surf_p1_avg)
print(roms_temp_p1_eid_coords_1m_from_surf.shape)
# 1 m above seafloor
roms_salt_p1_eid_coords_1m_above_seabed = regridder_rho2ctd_2020(roms_salt_1m_above_seafloor_p1_avg)
print(roms_temp_p1_eid_coords_1m_above_seabed.shape)

# Trim to the values we want
# Temperature
# 1m below surface
roms_temp_1m_from_surf_eidpoints = np.empty((lon_len, lat_len))
roms_temp_1m_from_surf_eidpoints[:36,:36] = roms_temp_p1_eid_coords_1m_from_surf[:36,:36]
roms_temp_1m_from_surf_eidpoints[36:59,36:59] = roms_temp_p1_eid_coords_1m_from_surf[37:60,37:60]
roms_temp_1m_from_surf_eidpoints[59:,59:] = roms_temp_p1_eid_coords_1m_from_surf[61:90,61:90]
# 1 m above seafloor
roms_temp_1m_above_seafloor_eidpoints = np.empty((lon_len, lat_len))
roms_temp_1m_above_seafloor_eidpoints[:36,:36] = roms_temp_p1_eid_coords_1m_above_seabed[:36,:36]
roms_temp_1m_above_seafloor_eidpoints[36:59,36:59] = roms_temp_p1_eid_coords_1m_above_seabed[37:60,37:60]
roms_temp_1m_above_seafloor_eidpoints[59:,59:] = roms_temp_p1_eid_coords_1m_above_seabed[61:90,61:90]
# SSC
# 1m below surface
roms_ssc_1m_from_surf_eidpoints = np.empty((lon_len, lat_len))
roms_ssc_1m_from_surf_eidpoints[:36,:36] = roms_ssc_p1_eid_coords_1m_from_surf[:36,:36]
roms_ssc_1m_from_surf_eidpoints[36:59,36:59] = roms_ssc_p1_eid_coords_1m_from_surf[37:60,37:60]
roms_ssc_1m_from_surf_eidpoints[59:,59:] = roms_ssc_p1_eid_coords_1m_from_surf[61:90,61:90]
# 1 m above seafloor
roms_ssc_1m_above_seafloor_eidpoints = np.empty((lon_len, lat_len))
roms_ssc_1m_above_seafloor_eidpoints[:36,:36] = roms_ssc_p1_eid_coords_1m_above_seabed[:36,:36]
roms_ssc_1m_above_seafloor_eidpoints[36:59,36:59] = roms_ssc_p1_eid_coords_1m_above_seabed[37:60,37:60]
roms_ssc_1m_above_seafloor_eidpoints[59:,59:] = roms_ssc_p1_eid_coords_1m_above_seabed[61:90,61:90]
# Salinity 
roms_salt_1m_from_surf_eidpoints = np.empty((lon_len, lat_len))
roms_salt_1m_from_surf_eidpoints[:36,:36] = roms_salt_p1_eid_coords_1m_from_surf[:36,:36]
roms_salt_1m_from_surf_eidpoints[36:59,36:59] = roms_salt_p1_eid_coords_1m_from_surf[37:60,37:60]
roms_salt_1m_from_surf_eidpoints[59:,59:] = roms_salt_p1_eid_coords_1m_from_surf[61:90,61:90]
# 1 m above seafloor
roms_salt_1m_above_seafloor_eidpoints = np.empty((lon_len, lat_len))
roms_salt_1m_above_seafloor_eidpoints[:36,:36] = roms_salt_p1_eid_coords_1m_above_seabed[:36,:36]
roms_salt_1m_above_seafloor_eidpoints[36:59,36:59] = roms_salt_p1_eid_coords_1m_above_seabed[37:60,37:60]
roms_salt_1m_above_seafloor_eidpoints[59:,59:] = roms_salt_p1_eid_coords_1m_above_seabed[61:90,61:90]

# Take the differences
# Since ROMS is 2D, need to loop through but only want diagonal points
# Make empty arrays to hold differences
# Temperature
roms_minus_eid_temp_1m_from_surf = np.empty(lon_len)
roms_minus_eid_temp_1m_above_seafloor = np.empty(lon_len)
roms_temp_1m_from_surf_eidpoints_only = np.empty(lon_len)
roms_temp_1m_above_seafloor_eidpoints_only = np.empty(lon_len)
# SSC
roms_minus_eid_ssc_1m_from_surf = np.empty(lon_len)
roms_minus_eid_ssc_1m_above_seafloor = np.empty(lon_len)
# Salinity
roms_salt_1m_from_surf_eidpoints_only = np.empty(lon_len)
roms_salt_1m_above_seafloor_eidpoints_only = np.empty(lon_len)
# Loop through lon
for n in range(lon_len):
    # Temperature 
    # Subtract the data from the observations and save to array 
    temp_diff_surf = roms_temp_1m_from_surf_eidpoints[n,n] - obs_temp_1m_from_surf_p1[n]
    temp_diff_seafloor = roms_temp_1m_above_seafloor_eidpoints[n,n] - obs_temp_1m_above_seafloor_p1[n]
    roms_temp_1m_from_surf_eidpoints_only[n] = roms_temp_1m_from_surf_eidpoints[n,n]
    roms_temp_1m_above_seafloor_eidpoints_only[n] = roms_temp_1m_above_seafloor_eidpoints[n,n]
    
    # Save to arrays 
    roms_minus_eid_temp_1m_from_surf[n] = temp_diff_surf
    roms_minus_eid_temp_1m_above_seafloor[n] = temp_diff_seafloor
    
    # SSC
    # Subtract the data from the observations and save to array 
    ssc_diff_surf = roms_ssc_1m_from_surf_eidpoints[n,n] - obs_ssc_1m_from_surf_p1[n]
    ssc_diff_seafloor = roms_ssc_1m_above_seafloor_eidpoints[n,n] - obs_ssc_1m_above_seafloor_p1[n]
    
    # Save to arrays 
    roms_minus_eid_ssc_1m_from_surf[n] = ssc_diff_surf
    roms_minus_eid_ssc_1m_above_seafloor[n] = ssc_diff_seafloor
    
    # Salinity
    roms_salt_1m_from_surf_eidpoints_only[n] = roms_salt_1m_from_surf_eidpoints[n,n]
    roms_salt_1m_above_seafloor_eidpoints_only[n] = roms_salt_1m_above_seafloor_eidpoints[n,n]

# Take the mean of these 
# Temperature
roms_minus_eid_temp_1m_from_surf_avg = np.nanmean(roms_minus_eid_temp_1m_from_surf)
roms_minus_eid_temp_1m_above_seafloor_avg = np.mean(roms_minus_eid_temp_1m_above_seafloor)
# SSC
roms_minus_eid_ssc_1m_from_surf_avg = np.nanmean(roms_minus_eid_ssc_1m_from_surf)
roms_minus_eid_ssc_1m_above_seafloor_avg = np.mean(roms_minus_eid_ssc_1m_above_seafloor)

# Print a bunch of stats
# Temperature
print('Temperature: ')
# Surface
print('Surface:')
print('Min Difference (C): ', np.nanmin(roms_minus_eid_temp_1m_from_surf))
print('Max Difference (C): ', np.nanmax(roms_minus_eid_temp_1m_from_surf))
print('Std. Dev. Difference (C): ', np.nanstd(roms_minus_eid_temp_1m_from_surf))
print('Mean Difference (C): ', roms_minus_eid_temp_1m_from_surf_avg)
# Bottom
print('Bottom:')
print('Min Difference (C): ', np.min(roms_minus_eid_temp_1m_above_seafloor))
print('Max Difference (C): ', np.max(roms_minus_eid_temp_1m_above_seafloor))
print('Std. Dev. Difference (C): ', np.std(roms_minus_eid_temp_1m_above_seafloor))
print('Mean Difference (C): ', roms_minus_eid_temp_1m_above_seafloor_avg)
# Net Mean
roms_minus_eid_temp_both_depths_avg = (roms_minus_eid_temp_1m_from_surf_avg+roms_minus_eid_temp_1m_from_surf_avg)/2
print('**Net Mean Difference (C)**: ', roms_minus_eid_temp_both_depths_avg)
# Mean values
print('Mean Eidam Obs Surf (C): ', np.nanmean(obs_temp_1m_from_surf_p1))
print('Mean Eidam Obs Bot (C): ', np.nanmean(obs_temp_1m_above_seafloor_p1))
print('Mean Eidam ROMS Surf (C): ', np.nanmean(roms_temp_1m_from_surf_eidpoints_only))
print('Mean Eidam ROMS Bot (C): ', np.nanmean(roms_temp_1m_above_seafloor_eidpoints_only))
# Standard deviations 
print('Std Dev Eidam Obs Surf (C): ', np.nanstd(obs_temp_1m_from_surf_p1))
print('Std Dev Eidam Obs Bot (C): ', np.std(obs_temp_1m_above_seafloor_p1))
print('Std Dev Eidam ROMS Surf (C): ', np.std(roms_temp_1m_from_surf_eidpoints_only))
print('Std Dev Eidam ROMS Bot (C): ', np.std(roms_temp_1m_above_seafloor_eidpoints_only))

# SSC
print('SSC: ')
# Surface
print('Surface:')
print('Min Difference (mg/L): ', np.nanmin(roms_minus_eid_ssc_1m_from_surf))
print('Max Difference (mg/L): ', np.nanmax(roms_minus_eid_ssc_1m_from_surf))
print('Std. Dev. Difference (mg/L): ', np.nanstd(roms_minus_eid_ssc_1m_from_surf))
print('Mean Difference (mg/L): ', roms_minus_eid_ssc_1m_from_surf_avg)
# Bottom
print('Bottom:')
print('Min Difference (mg/L): ', np.min(roms_minus_eid_ssc_1m_above_seafloor))
print('Max Difference (mg/L): ', np.max(roms_minus_eid_ssc_1m_above_seafloor))
print('Std. Dev. Difference (mg/L): ', np.std(roms_minus_eid_ssc_1m_above_seafloor))
print('Mean Difference (mg/L): ', roms_minus_eid_ssc_1m_above_seafloor_avg)
# Net Mean
roms_minus_eid_ssc_both_depths_avg = (roms_minus_eid_ssc_1m_from_surf_avg+roms_minus_eid_ssc_1m_from_surf_avg)/2
print('**Net Mean Difference (mg/L)**: ', roms_minus_eid_ssc_both_depths_avg)


# Salinity 
print('Salinity: ')
# Mean values
print('Mean Eidam Obs Surf (C): ', np.nanmean(obs_salt_1m_from_surf_p1))
print('Mean Eidam Obs Bot (C): ', np.nanmean(obs_salt_1m_above_seafloor_p1))
print('Mean Eidam ROMS Surf (C): ', np.nanmean(roms_salt_1m_from_surf_eidpoints_only))
print('Mean Eidam ROMS Bot (C): ', np.nanmean(roms_salt_1m_above_seafloor_eidpoints_only))
# Standard deviations 
print('Std Dev Eidam Obs Surf (C): ', np.nanstd(obs_salt_1m_from_surf_p1))
print('Std Dev Eidam Obs Bot (C): ', np.std(obs_salt_1m_above_seafloor_p1))
print('Std Dev Eidam ROMS Surf (C): ', np.std(roms_salt_1m_from_surf_eidpoints_only))
print('Std Dev Eidam ROMS Bot (C): ', np.std(roms_salt_1m_above_seafloor_eidpoints_only))


# Salinity differences in the region where the model is saltier (see figure 2)
# Make a list of the indices
salty_idx = [0, 18, 19, 20, 21, 25, 26, 27, 28]

# Get the ROMS value here
roms_salt_1m_from_surf_at_salty_pnts = roms_salt_1m_from_surf_eidpoints_only[salty_idx[:]]

# Get the obs value here
eid_salt_1m_from_surf_at_salty_pnts = obs_salt_1m_from_surf_p1[salty_idx[:]]

# Take the difference
roms_minus_eid_surf_salt_at_salty_pnts = roms_salt_1m_from_surf_at_salty_pnts - eid_salt_1m_from_surf_at_salty_pnts

# Print the mean difference 
roms_minus_eid_surf_salt_at_salty_pnts_mean = np.mean(roms_minus_eid_surf_salt_at_salty_pnts)
print('ROMS minus Eid salt mean: ', roms_minus_eid_surf_salt_at_salty_pnts_mean)

# Print the max difference 
roms_minus_eid_surf_salt_at_salty_pnts_max = np.max(roms_minus_eid_surf_salt_at_salty_pnts)
print('ROMS minus Eid salt max: ', roms_minus_eid_surf_salt_at_salty_pnts_max)

# Print the min difference 
roms_minus_eid_surf_salt_at_salty_pnts_min = np.min(roms_minus_eid_surf_salt_at_salty_pnts)
print('ROMS minus Eid salt min: ', roms_minus_eid_surf_salt_at_salty_pnts_min)

# Print the standard deviation of the difference 
roms_minus_eid_surf_salt_at_salty_pnts_std = np.std(roms_minus_eid_surf_salt_at_salty_pnts)
print('ROMS minus Eid salt std: ', roms_minus_eid_surf_salt_at_salty_pnts_std)


# Calculate and print the RMSE for the different variables
# Make a list of the values 
# Temperature 
temp_surf_obs = np.concatenate([
    obs_temp_1m_above_seafloor_p1[:36],
    obs_temp_1m_above_seafloor_p1[37:60],
    obs_temp_1m_above_seafloor_p1[61:90]])

temp_surf_roms = np.concatenate([
    roms_temp_1m_from_surf_on_eidam_points[:36],
    roms_temp_1m_from_surf_on_eidam_points[37:60],
    roms_temp_1m_from_surf_on_eidam_points[61:90]])

temp_bot_obs = np.concatenate([
    obs_temp_1m_above_seafloor_p1[:36],
    obs_temp_1m_above_seafloor_p1[37:60],
    obs_temp_1m_above_seafloor_p1[61:90]])

temp_bot_roms = np.concatenate([
    roms_temp_1m_above_seafloor_on_eidam_points[:36],
    roms_temp_1m_above_seafloor_on_eidam_points[37:60],
    roms_temp_1m_above_seafloor_on_eidam_points[61:90]])

# Salinity 
salt_surf_obs = np.concatenate([
    obs_salt_1m_from_surf_p1[:36],
    obs_salt_1m_from_surf_p1[37:60],
    obs_salt_1m_from_surf_p1[61:90]])
    
salt_surf_roms = np.concatenate([
    roms_salt_1m_from_surf_on_eidam_points[:36],
    roms_salt_1m_from_surf_on_eidam_points[37:60],
    roms_salt_1m_from_surf_on_eidam_points[61:90]])

salt_bot_obs = np.concatenate([
    obs_salt_1m_above_seafloor_p1[:36],
    obs_salt_1m_above_seafloor_p1[37:60],
    obs_salt_1m_above_seafloor_p1[61:90]])

salt_bot_roms = np.concatenate([
    roms_salt_1m_above_seafloor_on_eidam_points[:36],
    roms_salt_1m_above_seafloor_on_eidam_points[37:60],
    roms_salt_1m_above_seafloor_on_eidam_points[61:90]])

# SSC
ssc_surf_obs = np.concatenate([
    obs_ssc_1m_from_surf_p1[:36],
    obs_ssc_1m_from_surf_p1[37:60],
    obs_ssc_1m_from_surf_p1[61:90]])
    
ssc_surf_roms = np.concatenate([
    roms_ssc_1m_from_surf_on_eidam_points[:36],
    roms_ssc_1m_from_surf_on_eidam_points[37:60],
    roms_ssc_1m_from_surf_on_eidam_points[61:90]])

ssc_bot_obs = np.concatenate([
    obs_ssc_1m_above_seafloor_p1[:36],
    obs_ssc_1m_above_seafloor_p1[37:60],
    obs_ssc_1m_above_seafloor_p1[61:90]])

ssc_bot_roms = np.concatenate([
    roms_ssc_1m_above_seafloor_on_eidam_points[:36],
    roms_ssc_1m_above_seafloor_on_eidam_points[37:60],
    roms_ssc_1m_above_seafloor_on_eidam_points[61:90]])


# More SSC
print('SSC: ')
# Mean values
print('Mean Eidam Obs Surf (C): ', np.nanmean(ssc_surf_obs))
print('Mean Eidam Obs Bot (C): ', np.nanmean(ssc_bot_obs))
print('Mean Eidam ROMS Surf (C): ', np.nanmean(ssc_surf_roms*1000)) 
print('Mean Eidam ROMS Bot (C): ', np.nanmean(ssc_bot_roms*1000))
# Standard deviations 
print('Std Dev Eidam Obs Surf (C): ', np.nanstd(ssc_surf_obs))
print('Std Dev Eidam Obs Bot (C): ', np.std(ssc_bot_obs))
print('Std Dev Eidam ROMS Surf (C): ', np.std(ssc_surf_roms*1000))
print('Std Dev Eidam ROMS Bot (C): ', np.std(ssc_bot_roms*1000))
# Max
print('max Eidam obs surf (mg/L): ', np.nanmax(ssc_surf_obs))
print('max Eidam obs bot (mg/L): ', np.nanmax(ssc_bot_obs))

  

# Now calculate and print RMSE 
# Temperature 
surf_temp_rmse = sum((abs(temp_surf_roms-temp_surf_obs)**2)/len(temp_surf_obs))**0.5
print('surface temp rmse: ', surf_temp_rmse)
bot_temp_rmse = sum((abs(temp_bot_roms-temp_bot_obs)**2)/len(temp_bot_obs))**0.5
print('bottom temp rmse: ', bot_temp_rmse)

# Salinity
mask_nan_surf_salt = ~np.isnan(salt_surf_obs) & ~np.isnan(salt_surf_roms)
surf_salt_rmse = sum((abs(salt_surf_roms[mask_nan_surf_salt]-salt_surf_obs[mask_nan_surf_salt])**2)/len(salt_surf_obs[mask_nan_surf_salt]))**0.5
print('surface salt rmse: ', surf_salt_rmse)
bot_salt_rmse = sum((abs(salt_bot_roms-salt_bot_obs)**2)/len(salt_bot_obs))**0.5
print('bottom salt rmse: ', bot_salt_rmse)

# SSC
mask_nan_surf_ssc = ~np.isnan(ssc_surf_obs) & ~np.isnan(ssc_surf_roms)
surf_ssc_rmse = sum((abs(ssc_surf_roms[mask_nan_surf_ssc]*1000-ssc_surf_obs[mask_nan_surf_ssc])**2)/len(ssc_surf_obs[mask_nan_surf_ssc]))**0.5
print('surface ssc rmse: ', surf_ssc_rmse)
bot_ssc_rmse = sum((abs(ssc_bot_roms*1000-ssc_bot_obs)**2)/len(ssc_bot_obs))**0.5
print('bottom ssc rmse: ', bot_ssc_rmse)

# Calculate and print correlations, too
from scipy import stats
# Temperature 
surf_temp_cor = stats.pearsonr(temp_surf_obs, temp_surf_roms)
print('surface temp cor: ', surf_temp_cor)
bot_temp_cor = stats.pearsonr(temp_bot_obs, temp_bot_roms) 
print('bottom temp cor: ', bot_temp_cor)

# Salinity
surf_salt_cor = stats.pearsonr(salt_surf_obs[mask_nan_surf_salt], salt_surf_roms[mask_nan_surf_salt]) 
print('surface salt cor: ', surf_salt_cor)
bot_salt_cor = stats.pearsonr(salt_bot_obs, salt_bot_roms)
print('bottom salt cor: ', bot_salt_cor)

# SSC
surf_ssc_cor = stats.pearsonr(ssc_surf_obs[mask_nan_surf_ssc], ssc_surf_roms[mask_nan_surf_ssc]*1000) 
print('surface ssc cor: ', surf_ssc_cor)
bot_ssc_cor = stats.pearsonr(ssc_bot_obs, ssc_bot_roms*1000) 
print('bottom ssc cor: ', bot_ssc_cor)




# -------------------------------------------------------------------------------
# ---- Make a netcdf to hold the output data used for plotting 
# -------------------------------------------------------------------------------
# Make a temporary eta and xi 
eta_tmp = np.arange(0, 206, 1)
xi_tmp = np.arange(0, 346, 1)

# Set up the data
roms_temp_salt_ssc_at_eidam = xr.Dataset(
    data_vars=dict(
        roms_temp_1m_from_surf_avg=(['eta','xi'], roms_temp_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140].values),
        roms_temp_1m_above_seafloor_avg=(['eta','xi'], roms_temp_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140].values),
        roms_salt_1m_from_surf_avg=(['eta','xi'], roms_salt_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140].values),
        roms_salt_1m_above_seafloor_avg=(['eta','xi'], roms_salt_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140].values),
        roms_ssc_1m_from_surf_avg=(['eta','xi'], roms_ssc_1m_from_surf_p1_avg_wland_masked_trim[:,50:-140].values),
        roms_ssc_1m_above_seafloor_avg=(['eta','xi'], roms_ssc_1m_above_seafloor_p1_avg_wland_masked_trim[:,50:-140].values),
        lat=(['eta','xi'], lat_rho_trimmed[:,50:-140]),
        lon=(['eta','xi'], lon_rho_trimmed[:,50:-140])
        ),
    coords=dict(
        eta=('eta', eta_tmp),
        xi=('xi', xi_tmp), 
        ),
    attrs=dict(description='ROMS temperature (Celsius), salinity (PSU), and suspended sediment concentrations (SSC, kg/m3) averaged over time period of Eidam et al. in-situ observations'))
# Save to a netcdf
#roms_temp_salt_ssc_at_eidam.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Figure_data/roms_temp_salt_ssc_surf_bot.nc')




