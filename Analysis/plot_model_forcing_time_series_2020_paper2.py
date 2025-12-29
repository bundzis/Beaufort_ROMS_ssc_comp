#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:02:46 2023

@author: brun1463
"""

####################### Model Forcing Time Series ##############################
# The purpose of this script is to plot spatially-averaged time series of the
# different model forcings. This figure will mainly be used in the paper.
#
# Notes:
# - Right now, this script plots both wind and surface stress but the final
#   figure in the paper will likely have only one of these so whichever
#   ends up in the final plot will be in the first panel of the plot
# - This script needs to be run in base env 
# - This script has been updated to use the 2020 model run  for paper 2
################################################################################


# Load in the packages 
import numpy as np
import xarray as xr
import pandas as pd
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
#from matplotlib import transforms 
import cmocean
#import matplotlib.ticker as tick
#import matplotlib.patches as patches
#from matplotlib.colors import LinearSegmentedColormap

# Set the tick size for all plots
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# Prevent tick labels from overlapping
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12

# Load in the model grid
grid = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Grids/KakAKgrd_shelf_big010_smooth006.nc') 

# Load in some output
#output = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Output/dbsed0003/ocean_his_beaufort_shelf_2020_dbsed0003_0001.nc') 


# -------------------------------------------------
# ------ Plot 1: Ocean Bathymetry and Grid ------
# -------------------------------------------------
# Make a plot of the ocean bathymetry from 5 - 200 m
# and have every 16th grid line drawn, as well as the good
# land mask

# Make a plot of bathymetry with grid lines
# depths for the colorbar
fig1, ax1 = plt.subplots(figsize=(25,12)) # (15, 12) for horizontal colorbar; (20, 12) for vertical colorbar

# get rid of land
#noland_new_6 = np.ma.masked_where(grid.h[:,:].values > 0, grid.h.values, copy=True)

# determine spacing of contours
lev1 = np.arange(grid.h.values.min()-1, 200, 1)
cmap1 = cmocean.cm.deep
cmap1.set_under('darkgray')

lev2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# plot the bathymetry
#cs1 = ax1.contourf(grid.lon_rho.values, grid.lat_rho.values, output.bath[0,:,:].values*grid.mask_rho.values, lev1, cmap=cmap1, extend='both')
cs1 = ax1.contourf(grid.lon_rho.values, grid.lat_rho.values, grid.h[:,:].values*grid.mask_rho.values, lev1, cmap=cmap1, extend='both')
#cs1.cmap.set_under('darkgray')


# Try to plot grid lines...
# this puts the vertical lines through the longitudes over all latitudes 
ax1.plot(grid.lon_rho[:, ::16].values, grid.lat_rho[:, ::16].values, '-', color='k')

# this plots the vertical line on the right boundary of the grid
ax1.plot(grid.lon_rho[:, -1].values, grid.lat_rho[:, -1].values, '-', color='k')

# this plots the upper horizontal boundary of the grid
ax1.plot(grid.lon_rho[-1, :].values, grid.lat_rho[-1, :].values, '-', color='k')

for i in range(12):
    # this plots the horizontal lines (including the bottom boundary)
    ax1.plot(grid.lon_rho[(i*16), :].values, grid.lat_rho[(i*16), :].values, '-', color='k')


# add title
ax1.set_title('The Grid Cells (every 16th) in the Beaufort Sea', fontsize=30, y=1.08)
 
# format axes
ax1.set_xlabel('Longitude (degrees)', fontsize=30)
ax1.set_ylabel('Latitude (degrees)', fontsize=30)
#plt.grid(True)
 
# specify colorbar
#cbar6 = fig6.colorbar(cs9, label='Depth (meters)', orientation='horizontal')
cbar1 = fig1.colorbar(cs1, label='Depth (m)', orientation='vertical')
cbar1.set_label(label='Depth (meters)', size=30)



# -------------------------------------------------
# ------ Plot 2: Model Forcings ------
# -------------------------------------------------
# Make time series of model forcings: winds, currents,
# significant wave height (spatially-averaged), and
# total river water and sediment discharge
# Use the data fromt the forcing files used in the model
# run, not the most up-to-date ones 

# Read in the model forcings
# Paper 1
# =============================================================================
# # Winds
# wind_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/wind_forcing_file_beaufort_shelf_era5_2020_data001.nc')
# # Currents
# ubar_clm = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/Final_bryclm_conds/Attempt001/ubar_currents_clm_001.nc')
# vbar_clm = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/Final_bryclm_conds/Attempt001/vbar_currents_clm_002.nc')
# # Waves 
# wave_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/wave_forcing_file_kaktovik_shelf_ww3_2020_data002.nc')
# # Rivers
# # use river_forcing_file_kaktovik_shelf_radr_data_003.nc if using new model output
# river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/river_forcing_file_kaktovik_shelf_radr_data_2020_002.nc') 
# # Sea ice concentration 
# ice_data = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/sea_ice_concentration_forcing_file_beaufort_shelf_nsidc_2020_data001.nc')
# # Load in the surface stress
# sustr_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/sustr_forcing_file_kaktovik_shelf_hycom_data_2020_001.nc')
# svstr_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/svstr_forcing_file_kaktovik_shelf_hycom_data_2020_001.nc')
# 
# =============================================================================
# Paper 2
# Currents
ubar_clm = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/Final_bryclm_conds/Attempt001/ubar_currents_clm_001.nc')
vbar_clm = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/Final_bryclm_conds/Attempt001/vbar_currents_clm_002.nc')
# Waves 
wave_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/wave_forcing_file_kaktovik_shelf_cop_2020_data002.nc')
# Rivers
# use river_forcing_file_kaktovik_shelf_radr_data_003.nc if using new model output
river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/Rivers/river_forcing_file_beaufort_shelf_10rivs_13seabed_blaskey_data_sagDSS3_rating_001.nc') 
# Sea ice concentration 
ice_data = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/sea_ice_concentration_forcing_file_beaufort_shelf_nsidc_2020_data001.nc')
# Load in the surface stress
sustr_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/sustr_forcing_file_kaktovik_shelf_hycom_data_2020_001.nc')
svstr_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2020/Model_Inputs/Forcing_files/svstr_forcing_file_kaktovik_shelf_hycom_data_2020_001.nc')



# Convert time
#time_data_wind = pd.to_datetime(wind_frc.wind_time.values+86400, origin=pd.datetime(1999,12,31), unit='s') # or try from datetime import datetime then jsut use datetime
time_data_wave = pd.to_datetime(wave_frc.wave_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')
time_data_cur = pd.to_datetime(ubar_clm.v2d_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')
time_data_riv = river_frc.river_time.values
time_data_ice = pd.to_datetime(ice_data.ice_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')
time_data_sustr = pd.to_datetime(sustr_frc.sms_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')
time_data_svstr = pd.to_datetime(svstr_frc.sms_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')
#time_data_riv = pd.to_datetime(river_frc.river_time.values+86400, origin=pd.datetime(1999,12,31), unit='s')

# Pre-process - take spatial averages
# Winds 
#uwind_avg = wind_frc.Uwind.mean(dim=('lat', 'lon'))
#vwind_avg = wind_frc.Vwind.mean(dim=('lat', 'lon'))
# Currents
ucur_avg = ubar_clm.ubar.mean(dim=('eta_u', 'xi_u'))
vcur_avg = vbar_clm.vbar.mean(dim=('eta_v', 'xi_v'))
# Significant Wave Height 
swh_avg = wave_frc.Hwave.mean(dim=('eta_rho', 'xi_rho'))
# Surface stress
sustr_avg = sustr_frc.sustr.mean(dim=('eta_u', 'xi_u'))
svstr_avg = svstr_frc.svstr.mean(dim=('eta_v', 'xi_v'))

# Pre-process - add together river inputs
water_dis_tot = river_frc.river_transport.sum(dim='river')
water_dis_col = river_frc.river_transport[:,1:7].sum(dim='river')
#water_dis_kuk = river_frc.river_transport[:,12].values
water_dis_kup = river_frc.river_transport[:,12].values + river_frc.river_transport[:,13].values
# Add together all of the sediment classes
riv_all_ssc = river_frc.river_mud_01 + river_frc.river_mud_02 + river_frc.river_sand_01 + river_frc.river_sand_02 + river_frc.river_sand_03
water_sed_tot = riv_all_ssc.sum(dim='river')
water_sed_col = riv_all_ssc[:,1:7].sum(dim='river')
#water_sed_kuk = riv_all_ssc[:,12]
water_sed_kup = riv_all_ssc[:,12] + riv_all_ssc[:,13]
# Mulitply this by the water discharge to get river sediment discharge 
water_sed_tot_kgs = water_sed_tot*water_dis_tot
water_sed_col_kgs = water_sed_col*water_dis_col
#water_sed_kuk_kgs = water_sed_kuk*water_dis_kuk
water_sed_kup_kgs = water_sed_kup*water_dis_kup
# Take 2 for river sediment load 
# I think this one below is more correct so use this instead for plots 
water_sed_tot_kgs_2 = (river_frc.river_transport*riv_all_ssc).sum(dim='river')
water_sed_col_kgs_2 = (river_frc.river_transport[:,1:7]*riv_all_ssc[:,1:7]).sum(dim='river')
#water_sed_kuk_kgs_2 = (river_frc.river_transport[:,12]*riv_all_ssc[:,12])
water_sed_kup_kgs_2 = ((river_frc.river_transport[:,12]+river_frc.river_transport[:,13])*(riv_all_ssc[:,12]+riv_all_ssc[:,13]))

# Pre-process - take spatial averaged of sea ice
# Find the max, min, median
mean_cover = ice_data.sea_ice_concentration.mean(dim=('eta_rho', 'xi_rho'))


# Make the figure 
fig1 = plt.figure(figsize=(32,30)) #(horizontal, vertical) (32,38) (32,30)
#fig1 = plt.figure(figsize=(32,16)) #(horizontal, vertical) 

# Set height ratios for subplots
gs = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,1,1])
#gs = gridspec.GridSpec(7, 1, height_ratios=[1,1,1,1,1,1,1])
#gs = gridspec.GridSpec(7, 1, height_ratios=[1.5,1.5,1,1,1,1,1.5])
#gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])

# Winds
ax0 = plt.subplot(gs[0])
#line0, = ax0.plot(time_data_wind, uwind_avg, color='r', label= '$\\bf{Eastward}$', linewidth=8) #time_data_wind[:938] for July,
#line01, = ax0.plot(time_data_wind, vwind_avg, color='b', label='$\\bf{Northward}$', linewidth=8)
ax0.set_ylabel('Wind \nSpeed \n(m/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax0.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax0.get_xticklabels(), visible=False)
xticks1 = ax0.xaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Currents
ax1 = plt.subplot(gs[1], sharex= ax0)
line1, = ax1.plot(time_data_cur, ucur_avg, color='orangered', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_cur[:313] for July 
line11, = ax1.plot(time_data_cur, vcur_avg, color='cornflowerblue', label='$\\bf{Across-Shore}$', linewidth=8)
ax1.set_ylabel('Larger-Scale \nCurrent \nSpeed \n(m/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=118, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax1.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax1.get_xticklabels(), visible=False)
yticks1 = ax1.yaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Waves
# shared axis X
ax2 = plt.subplot(gs[2], sharex = ax0)
line2, = ax2.plot(time_data_wave, swh_avg, color='purple', label='$\\bf{Significant Wave Height}$', linewidth=8) #time_data_wave[:313] for July 
#ax2.axhspan(ax2.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
#ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax2.get_xticklabels(), visible=False)
ax2.set_ylabel('Significant \nWave \nHeight \n(m)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=130, va='center')
# remove last tick label for the second subplot
yticks2 = ax2.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River water discharge
# shared axis X
ax3 = plt.subplot(gs[3], sharex = ax0)
line3, = ax3.plot(time_data_riv, water_dis_tot, color='deepskyblue', label='$\\bf{River Water Discharge}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax3.get_xticklabels(), visible=False)
#ax3.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax3.set_ylabel('Water \nDischarge \n(m\u00b3/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# remove last tick label for the second subplot
yticks3 = ax3.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River sediment discharge
# shared axis X
ax4 = plt.subplot(gs[4], sharex = ax0)
line4, = ax4.plot(time_data_riv, water_sed_tot_kgs, color='peru', label='$\\bf{River Sediment Load}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax4.get_xticklabels(), visible=False)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax4.set_ylabel('River \nSediment \nLoad \n(kg/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=110, va='center')
# remove last tick label for the second subplot
yticks4 = ax4.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# Sea ice concentration 
# shared axis X
ax5 = plt.subplot(gs[5], sharex = ax0)
line5, = ax5.plot(time_data_ice, mean_cover, color='green', label='$\\bf{Sea Ice Concentration}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax5.get_xticklabels(), visible=True)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax5.set_ylabel('Sea Ice \nConc. \n(fraction \nof grid cell)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=115, va='center')
# remove last tick label for the second subplot
yticks5 = ax5.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# =============================================================================
# # Surface stress
# ax6 = plt.subplot(gs[6], sharex = ax0)
# line6, = ax6.plot(time_data_sustr, sustr_avg, color='salmon', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_wind[:938] for July,
# line61, = ax6.plot(time_data_svstr, svstr_avg, color='dodgerblue', label='$\\bf{Across-Shore}$', linewidth=8)
# ax6.set_ylabel('Surface \nStress \n(N/m\u00b2)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# #ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
# ax6.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
# plt.setp(ax6.get_xticklabels(), visible=True)
# xticks1 = ax6.xaxis.get_major_ticks()
# #ax0.set_xlabel('Time', fontsize=30)
# =============================================================================

# put legend on first subplot
#ax0.legend((line0, line1), ('red line', 'blue line'))
ax0.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
ax1.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax2.legend(fontsize=40)
#ax3.legend(fontsize=40)
#ax4.legend(fontsize=40)
#ax3.set_xlabel('Month in 2019')
#ax6.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
plt.xlabel('\nDate', fontsize=40, fontweight='bold')

# Add text labels for the panels (a, b, c, etc.)
# Use these if there are 6 subplots
plt.text(0.135, 0.849, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.727, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.595, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.47, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.345, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.22, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
# Use these if there are 7 subplots
#plt.text(0.135, 0.855, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.742, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.638, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.528, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.420, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.310, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.203, 'g)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)

#fig1.set_tight_layout()

plt.tight_layout


# remove vertical gap between subplots
#plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.08)
#plt.show()



# -------------------------------------------------
# ------ Plot 3: Mask Nudged and >60 m Depth ------
# -------------------------------------------------
# Make a mask that masks the nudged areas and regions the are 
# deeper than 60 meters deep

# First just mask areas deeper than 60 m
# Make a function to mask the data
def masked_array(data, threshold):
    return (data <= threshold).astype(int)

# Call the function to make the mask
depth_mask_rho = masked_array(grid.h.values, 60)

# Plot to see if this worked
fig7, ax7 = plt.subplots(figsize=(25,12))
lev7 = [0,1]
# Plot the mask
cs7 = ax7.contourf(grid.lon_rho.values, grid.lat_rho.values, depth_mask_rho, lev7, extend='both')
ax7.set_title('The Grid Masked Depths > 60 m', fontsize=30, y=1.08)
ax7.set_xlabel('Longitude (degrees)', fontsize=30)
ax7.set_ylabel('Latitude (degrees)', fontsize=30)
cbar7 = fig7.colorbar(cs7, label='Mask', orientation='vertical')
cbar7.set_label(label='Mask', size=30)

# Now manuallly edit the mask so that the nudged areas are 0 too
# Set the number of cells in the sponge on each open boundary
c_west = 36
c_north = 45
c_east = 36

# Make a copy of the depth mask to work with
depth_nudge_mask_rho = depth_mask_rho.copy()

# Manually change the values  [eta,xi]
depth_nudge_mask_rho[-c_north:,:] = 0
depth_nudge_mask_rho[:,:c_west] = 0
depth_nudge_mask_rho[:,-c_east:] = 0

# Plot to see if this worked
fig8, ax8 = plt.subplots(figsize=(25,12))
lev8 = [0,1]
# Plot the mask
cs8 = ax8.contourf(grid.lon_rho.values, grid.lat_rho.values, depth_nudge_mask_rho, lev8, extend='both')
ax8.set_title('The Grid Masked Depths > 60 m & Nudged', fontsize=30, y=1.08)
ax8.set_xlabel('Longitude (degrees)', fontsize=30)
ax8.set_ylabel('Latitude (degrees)', fontsize=30)
cbar8 = fig8.colorbar(cs8, label='Mask', orientation='vertical')
cbar8.set_label(label='Mask', size=30)


# Now plot the bathymetry with this mask applied
fig9, ax9 = plt.subplots(figsize=(25,12)) # (15, 12) for horizontal colorbar; (20, 12) for vertical colorbar
# determine spacing of contours
lev9 = np.arange(grid.h.values.min()-1, 60, 1)
cmap9 = cmocean.cm.deep
cmap9.set_under('darkgray')

# plot the bathymetry
#cs9 = ax9.contourf(grid.lon_rho.values, grid.lat_rho.values, output.bath[0,:,:].values*depth_nudge_mask_rho*grid.mask_rho.values, lev9, cmap=cmap9, extend='both')
cs9 = ax9.contourf(grid.lon_rho.values, grid.lat_rho.values, grid.h[:,:].values*depth_nudge_mask_rho*grid.mask_rho.values, lev9, cmap=cmap9, extend='both')
#cs1.cmap.set_under('darkgray')

# add title
ax9.set_title('The Masked Bathymetry', fontsize=30, y=1.08)
 
# format axes
ax9.set_xlabel('Longitude (degrees)', fontsize=30)
ax9.set_ylabel('Latitude (degrees)', fontsize=30)
#plt.grid(True)
 
# specify colorbar
#cbar6 = fig6.colorbar(cs9, label='Depth (meters)', orientation='horizontal')
cbar9 = fig9.colorbar(cs9, label='Depth (m)', orientation='vertical')
cbar9.set_label(label='Depth (meters)', size=30)



# --------------------------------------------------------------------------------
# ------ Plot 4: Mask Nudged and >60 m Depth Spatially-Averaged Time Series ------
# --------------------------------------------------------------------------------
# Make another mask where the west adn east nudged areas are ingored and depths 
# greater than 60 m are ignored
# Now apply this mask to the spatial-averaging of time series to take out these 
# regions from averages

# Does it make more sense for mask to be 0 or nan for the math?
#nudge_mask = np.full_like(depth_nudge_mask, fill_value=1)
# Call the function to make the mask
nudge_mask_rho = masked_array(grid.h.values, 60)
#nudge_mask[-c_north:,:] = 0
nudge_mask_rho[:,:c_west] = 0
nudge_mask_rho[:,-c_east:] = 0

# Plot to see if this worked
fig10, ax10 = plt.subplots(figsize=(25,12))
lev10 = [0,1]
# Plot the mask
cs10 = ax10.contourf(grid.lon_rho.values, grid.lat_rho.values, nudge_mask_rho, lev10, extend='both')
ax10.set_title('The Grid Masked Depths > 60 m & Nudged', fontsize=30, y=1.08)
ax10.set_xlabel('Longitude (degrees)', fontsize=30)
ax10.set_ylabel('Latitude (degrees)', fontsize=30)
cbar10 = fig10.colorbar(cs10, label='Mask', orientation='vertical')
cbar10.set_label(label='Mask', size=30)

# Now plot the bathymetry with this mask applied
fig11, ax11 = plt.subplots(figsize=(25,12)) # (15, 12) for horizontal colorbar; (20, 12) for vertical colorbar
# determine spacing of contours
lev11 = np.arange(grid.h.values.min()-1, 60, 1)
cmap11 = cmocean.cm.deep
cmap11.set_under('darkgray')

# plot the bathymetry
#cs11 = ax11.contourf(grid.lon_rho.values, grid.lat_rho.values, output.bath[0,:,:].values*nudge_mask_rho*grid.mask_rho.values, lev11, cmap=cmap11, extend='both')
cs11 = ax11.contourf(grid.lon_rho.values, grid.lat_rho.values, grid.h[:,:].values*nudge_mask_rho*grid.mask_rho.values, lev11, cmap=cmap11, extend='both')

#cs1.cmap.set_under('darkgray')

# add title
ax11.set_title('The Masked Bathymetry', fontsize=30, y=1.08)
 
# format axes
ax11.set_xlabel('Longitude (degrees)', fontsize=30)
ax11.set_ylabel('Latitude (degrees)', fontsize=30)
#plt.grid(True)
 
# specify colorbar
#cbar6 = fig6.colorbar(cs9, label='Depth (meters)', orientation='horizontal')
cbar11 = fig11.colorbar(cs11, label='Depth (m)', orientation='vertical')
cbar11.set_label(label='Depth (meters)', size=30)


# *****So we will go with this last mask for alla anlysis and plots*****
# --------------------------------------------------------------------------------
# ------ Plot 5: Masked  Spatially-Averaged Time Series ------
# --------------------------------------------------------------------------------
# Now apply this mask to the spatial-averaging of time series to take out these 
# regions from averages
# Does it make more sense for mask to be 0 or nan for the math?

# For the math, set all the regions that are 0 to nan in the new mask
nudge_mask_rho_nan = np.copy(nudge_mask_rho)
nudge_mask_rho_nan = np.where(nudge_mask_rho_nan != 0, nudge_mask_rho_nan, np.nan)

# Plot to see if this worked
fig12, ax12 = plt.subplots(figsize=(25,12))
lev12 = [0,1,3]
# Plot the mask
cs12 = ax12.contourf(grid.lon_rho.values, grid.lat_rho.values, nudge_mask_rho_nan, lev12, extend='both')
ax12.set_title('The Grid Masked Depths > 60 m & Nudged Nan', fontsize=30, y=1.08)
ax12.set_xlabel('Longitude (degrees)', fontsize=30)
ax12.set_ylabel('Latitude (degrees)', fontsize=30)
cbar12 = fig12.colorbar(cs12, label='Mask', orientation='vertical')
cbar12.set_label(label='Mask', size=30)


# Make the same masks but for uv points
nudge_mask_u = nudge_mask_rho[:,1:]
nudge_mask_v = nudge_mask_rho[1:,:]
nudge_mask_u_nan = nudge_mask_rho_nan[:,1:]
nudge_mask_v_nan = nudge_mask_rho_nan[1:,:]


# Now  redo the calculations for spatial averaging by first multiplying by this 
# nan mask, then taking the spatial average so that these regions will be ignored 
# Pre-process - take spatial averages but first multiply by the mask
# Currents, Waves, surface stress (since time is same length for all these files)
# Make empty arrays to hold the version multiplied by mask
# Currents
ubar_masked = np.empty_like((ubar_clm.ubar))
vbar_masked = np.empty_like((vbar_clm.vbar))
# Waves
swh_masked = np.empty_like((wave_frc.Hwave))
# Surface stress
sustr_masked = np.empty_like((sustr_frc.sustr))
svstr_masked = np.empty_like((svstr_frc.svstr))

for ii in range(len(ubar_clm.v2d_time)):
    # Currents
    ubar_masked[ii,:,:] = ubar_clm.ubar[ii,:,:] * nudge_mask_u_nan
    vbar_masked[ii,:,:] = vbar_clm.vbar[ii,:,:] * nudge_mask_v_nan
    # Waves
    #swh_masked[ii,:,:] = wave_frc.Hwave[ii,:,:] * nudge_mask_rho_nan
    # Surface stress 
    sustr_masked[ii,:,:] = sustr_frc.sustr[ii,:,:] * nudge_mask_u_nan
    svstr_masked[ii,:,:,] = svstr_frc.svstr[ii,:,:] * nudge_mask_v_nan
    
# Do waves in a separate loop since they have different time frequency 
for iii in range(len(time_data_wave)):
    # Waves
    swh_masked[iii,:,:] = wave_frc.Hwave[iii,:,:] * nudge_mask_rho_nan
    
# Now take the spatial average
ubar_avg_masked = np.nanmean(ubar_masked, axis=(1,2))
vbar_avg_masked = np.nanmean(vbar_masked, axis=(1,2))
# Significant Wave Height 
swh_avg_masked = np.nanmean(swh_masked, axis=(1,2))
# Surface stress
sustr_avg_masked = np.nanmean(sustr_masked, axis=(1,2))
svstr_avg_masked = np.nanmean(svstr_masked, axis=(1,2))

# Pre-process - take spatial averaged of sea ice
# Multiply by nan mask first 
# Make an empty array to hold masked version
sea_ice_concentration_masked = np.empty_like((ice_data.sea_ice_concentration))
for iii in range(len(ice_data.ice_time)):
    sea_ice_concentration_masked[iii,:,:] = ice_data.sea_ice_concentration[iii,:,:] * nudge_mask_rho_nan
    
# Find the mean
mean_cover_masked = np.nanmean(sea_ice_concentration_masked, axis=(1,2))

# Now plot all of these masked versions
# Make the figure 
fig13 = plt.figure(figsize=(32,30)) #(horizontal, vertical) (32,38) (32,30)
#fig13.suptitle('Masked Model Forcing Time Series', fontsize=40)

# Set height ratios for subplots
gs2 = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,1,1])

# Winds
ax13 = plt.subplot(gs2[0])
#line0, = ax13.plot(time_data_wind, uwind_avg, color='r', label= '$\\bf{Eastward}$', linewidth=8) #time_data_wind[:938] for July,
#line01, = ax13.plot(time_data_wind, vwind_avg, color='b', label='$\\bf{Northward}$', linewidth=8)
ax13.set_ylabel('Wind \nSpeed \n(m/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax13.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax13.get_xticklabels(), visible=False)
xticks13 = ax13.xaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Currents
ax14 = plt.subplot(gs2[1], sharex= ax13)
line1, = ax14.plot(time_data_cur, ubar_avg_masked*100, color='orangered', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_cur[:313] for July 
line11, = ax14.plot(time_data_cur, vbar_avg_masked*100, color='cornflowerblue', label='$\\bf{Across-Shore}$', linewidth=8)
ax14.set_ylabel('Larger-Scale \nCurrent \nSpeed \n(cm/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=118, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax14.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax14.get_xticklabels(), visible=False)
yticks14 = ax14.yaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Waves
# shared axis X
ax15 = plt.subplot(gs2[2], sharex = ax13)
line2, = ax15.plot(time_data_wave, swh_avg_masked, color='purple', label='$\\bf{Significant Wave Height}$', linewidth=8) #time_data_wave[:313] for July 
#ax2.axhspan(ax2.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
#ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax15.get_xticklabels(), visible=False)
ax15.set_ylabel('Significant \nWave \nHeight \n(m)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=130, va='center')
# remove last tick label for the second subplot
yticks15 = ax15.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River water discharge
# shared axis X
ax16 = plt.subplot(gs2[3], sharex = ax13)
line3, = ax16.plot(time_data_riv, water_dis_tot, color='deepskyblue', label='$\\bf{Total}$', linewidth=8) #time_data_riv[:40] for July 
# Colville
ax16.plot(time_data_riv, water_dis_col, label='$\\bf{Colville}$', linewidth=5, 
            color='m')
# Kukpuk 
ax16.plot(time_data_riv, water_dis_kup, label='$\\bf{Kuparuk}$', linewidth=5,
            color='peru')
plt.setp(ax16.get_xticklabels(), visible=False)
#ax3.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax16.set_ylabel('Water \nDischarge \n(m\u00b3/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# remove last tick label for the second subplot
yticks16 = ax16.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River sediment discharge
# shared axis X
ax17 = plt.subplot(gs2[4], sharex = ax13)
line4, = ax17.plot(time_data_riv, water_sed_tot_kgs_2, color='deepskyblue', label='$\\bf{Total}$', linewidth=8) #time_data_riv[:40] for July 
# Colville
ax17.plot(time_data_riv, water_sed_col_kgs_2, label='$\\bf{Colville}$', linewidth=5,
            color='m')
# Kukpuk 
ax17.plot(time_data_riv, water_sed_kup_kgs_2, label='$\\bf{Kuparuk}$', linewidth=5, 
            color='peru')
plt.setp(ax17.get_xticklabels(), visible=False)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax17.set_ylabel('River \nSediment \nLoad \n(kg/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=110, va='center')
# remove last tick label for the second subplot
yticks17 = ax17.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# Sea ice concentration 
# shared axis X
ax18 = plt.subplot(gs2[5], sharex = ax13)
line5, = ax18.plot(time_data_ice, mean_cover_masked, color='green', label='$\\bf{Sea Ice Concentration}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax18.get_xticklabels(), visible=True)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax18.set_ylabel('Sea Ice \nConc. \n(fraction \nof grid cell)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=115, va='center')
# remove last tick label for the second subplot
yticks18 = ax18.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# =============================================================================
# # Surface stress
# ax6 = plt.subplot(gs[6], sharex = ax0)
# line6, = ax6.plot(time_data_sustr, sustr_avg, color='salmon', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_wind[:938] for July,
# line61, = ax6.plot(time_data_svstr, svstr_avg, color='dodgerblue', label='$\\bf{Across-Shore}$', linewidth=8)
# ax6.set_ylabel('Surface \nStress \n(N/m\u00b2)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# #ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
# ax6.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
# plt.setp(ax6.get_xticklabels(), visible=True)
# xticks1 = ax6.xaxis.get_major_ticks()
# #ax0.set_xlabel('Time', fontsize=30)
# =============================================================================

# put legend on first subplot
#ax0.legend((line0, line1), ('red line', 'blue line'))
ax13.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
ax14.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax2.legend(fontsize=40)
ax16.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
ax17.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax3.legend(fontsize=40)
#ax4.legend(fontsize=40)
#ax3.set_xlabel('Month in 2019')
#ax6.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
plt.xlabel('\nDate', fontsize=40, fontweight='bold')

# Add text labels for the panels (a, b, c, etc.)
# Use these if there are 6 subplots
plt.text(0.135, 0.849, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.727, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.595, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.47, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.345, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.22, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
# Use these if there are 7 subplots
#plt.text(0.135, 0.855, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.742, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.638, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.528, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.420, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.310, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.203, 'g)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)

# Shade time periods of interest
# =============================================================================
# # Moccasin for periods looked at and plum for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# =============================================================================
# =============================================================================
# # Plum for periods looked at and moccasin for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax13.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax13.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax14.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax14.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax15.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax15.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax16.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax16.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax17.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax17.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax18.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax18.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# 
# =============================================================================

#fig1.set_tight_layout()

plt.tight_layout


# remove vertical gap between subplots
#plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.08)
#plt.show()


# Calcualte and print the total amount of sediment (kg) 
# input to the shelf over the run
# Take the kg/s values for total and multiply it by dt, then add up 
# dt is daily = 86400 seconds 
water_sed_tot_kg = np.sum(water_sed_tot_kgs_2*86400)
print('total kg sediment delivered: ', water_sed_tot_kg)

# Print the maximum total sediment load 
print('Maximum total sediment load (kg/s): ', np.max(water_sed_tot_kgs_2))

# Print min Colville sediment conc 
print('min Colville Sediment conc (kg/m3): ', np.min(water_sed_col))

# Print max Colville sediment conc
print('ax Colville Sediment conc (kg/m3): ', np.max(water_sed_col))


# --------------------------------------------------------------------------------
# ------ Plot 6: Masked  Spatially-Averaged Time Series ------
# --------------------------------------------------------------------------------
# Same as above but trimmed to only plot the time that the model actually runs for
# since the model ends on October 29, hour 4

# Make the figure 
fig14 = plt.figure(figsize=(32,30)) #(horizontal, vertical) (32,38) (32,30)
#fig13.suptitle('Masked Model Forcing Time Series', fontsize=40)

# Set height ratios for subplots
gs3 = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,1,1])

# Winds
ax19 = plt.subplot(gs3[0])
#line6, = ax19.plot(time_data_wind[:-93], uwind_avg[:-93], color='r', label= '$\\bf{Eastward}$', linewidth=8) #time_data_wind[:938] for July,
#line61, = ax19.plot(time_data_wind[:-93], vwind_avg[:-93], color='b', label='$\\bf{Northward}$', linewidth=8)
ax19.set_ylabel('Wind \nSpeed \n(m/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax19.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax19.get_xticklabels(), visible=False)
xticks19 = ax19.xaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Currents
ax20 = plt.subplot(gs3[1], sharex= ax19)
line7, = ax20.plot(time_data_cur[:-31], ubar_avg_masked[:-31]*100, color='orangered', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_cur[:313] for July 
line71, = ax20.plot(time_data_cur[:-31], vbar_avg_masked[:-31]*100, color='cornflowerblue', label='$\\bf{Across-Shore}$', linewidth=8)
ax20.set_ylabel('Larger-Scale \nCurrent \nSpeed \n(cm/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=118, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax20.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax20.get_xticklabels(), visible=False)
yticks20 = ax20.yaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Waves
# shared axis X
ax21 = plt.subplot(gs3[2], sharex = ax19)
line8, = ax21.plot(time_data_wave[:-117], swh_avg_masked[:-117], color='purple', label='$\\bf{Significant Wave Height}$', linewidth=8) #time_data_wave[:313] for July 
#ax2.axhspan(ax2.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
#ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax21.get_xticklabels(), visible=False)
ax21.set_ylabel('Significant \nWave \nHeight \n(m)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=130, va='center')
# remove last tick label for the second subplot
yticks21 = ax21.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River water discharge
# shared axis X
ax22 = plt.subplot(gs3[3], sharex = ax19)
line9, = ax22.plot(time_data_riv[:-4], water_dis_tot[:-4], color='deepskyblue', label='$\\bf{Total}$', linewidth=8) #time_data_riv[:40] for July 
# Colville
ax22.plot(time_data_riv[:-4], water_dis_col[:-4], label='$\\bf{Colville}$', linewidth=5, 
            color='m')
# Kukpuk 
ax22.plot(time_data_riv[:-4], water_dis_kup[:-4], label='$\\bf{Kuparuk}$', linewidth=5,
            color='peru')
plt.setp(ax22.get_xticklabels(), visible=False)
#ax3.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax22.set_ylabel('Water \nDischarge \n(m\u00b3/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# remove last tick label for the second subplot
yticks22 = ax22.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River sediment discharge
# shared axis X
ax23 = plt.subplot(gs3[4], sharex = ax19)
line10, = ax23.plot(time_data_riv[:-4], water_sed_tot_kgs_2[:-4], color='deepskyblue', label='$\\bf{Total}$', linewidth=8) #time_data_riv[:40] for July 
# Colville
ax23.plot(time_data_riv[:-4], water_sed_col_kgs_2[:-4], label='$\\bf{Colville}$', linewidth=5,
            color='m')
# Kukpuk 
ax23.plot(time_data_riv[:-4], water_sed_kup_kgs_2[:-4], label='$\\bf{Kuparuk}$', linewidth=5, 
            color='peru')
plt.setp(ax23.get_xticklabels(), visible=False)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax23.set_ylabel('River \nSediment \nLoad \n(kg/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=110, va='center')
# remove last tick label for the second subplot
yticks23 = ax23.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# Sea ice concentration 
# shared axis X
ax24 = plt.subplot(gs3[5], sharex = ax19)
line11, = ax24.plot(time_data_ice[:-4], mean_cover_masked[:-4], color='green', label='$\\bf{Sea Ice Concentration}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax24.get_xticklabels(), visible=True)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax24.set_ylabel('Sea Ice \nConc. \n(fraction \nof grid cell)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=115, va='center')
# remove last tick label for the second subplot
yticks24 = ax24.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# =============================================================================
# # Surface stress
# ax6 = plt.subplot(gs[6], sharex = ax0)
# line6, = ax6.plot(time_data_sustr, sustr_avg, color='salmon', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_wind[:938] for July,
# line61, = ax6.plot(time_data_svstr, svstr_avg, color='dodgerblue', label='$\\bf{Across-Shore}$', linewidth=8)
# ax6.set_ylabel('Surface \nStress \n(N/m\u00b2)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# #ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
# ax6.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
# plt.setp(ax6.get_xticklabels(), visible=True)
# xticks1 = ax6.xaxis.get_major_ticks()
# #ax0.set_xlabel('Time', fontsize=30)
# =============================================================================

# put legend on first subplot
#ax0.legend((line0, line1), ('red line', 'blue line'))
ax19.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
ax20.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax2.legend(fontsize=40)
ax22.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
ax23.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax3.legend(fontsize=40)
#ax4.legend(fontsize=40)
#ax3.set_xlabel('Month in 2019')
#ax6.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
plt.xlabel('\nDate', fontsize=40, fontweight='bold')

# Add text labels for the panels (a, b, c, etc.)
# Use these if there are 6 subplots
plt.text(0.135, 0.849, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.727, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.595, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.47, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.345, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.22, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
# Use these if there are 7 subplots
#plt.text(0.135, 0.855, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.742, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.638, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.528, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.420, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.310, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.203, 'g)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)

# Shade time periods of interest
# =============================================================================
# # Moccasin for periods looked at and plum for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# =============================================================================
# =============================================================================
# # Plum for periods looked at and moccasin for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax13.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax13.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax14.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax14.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax15.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax15.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax16.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax16.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax17.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax17.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax18.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax18.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# 
# =============================================================================

#fig1.set_tight_layout()

plt.tight_layout


# remove vertical gap between subplots
#plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.08)
#plt.show()



# --------------------------------------------------------------------------------
# ------ Plot 7: Masked  Spatially-Averaged Time Series ------
# --------------------------------------------------------------------------------
# Same as above but trimmed to only plot the time that the model actually runs for
# since the model ends on October 29, hour 4

# Make the figure 
fig15 = plt.figure(figsize=(32,30)) #(horizontal, vertical) (32,38) (32,30)
#fig13.suptitle('Masked Model Forcing Time Series', fontsize=40)

# Set height ratios for subplots
gs3 = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,1,1])

# =============================================================================
# # Winds
# ax19 = plt.subplot(gs3[0])
# line6, = ax19.plot(time_data_wind[:-93], uwind_avg[:-93], color='r', label= '$\\bf{Eastward}$', linewidth=8) #time_data_wind[:938] for July,
# line61, = ax19.plot(time_data_wind[:-93], vwind_avg[:-93], color='b', label='$\\bf{Northward}$', linewidth=8)
# ax19.set_ylabel('Wind \nSpeed \n(m/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# #ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
# ax19.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
# plt.setp(ax19.get_xticklabels(), visible=False)
# xticks19 = ax19.xaxis.get_major_ticks()
# #ax0.set_xlabel('Time', fontsize=30)
# =============================================================================

# Surface stress
ax19 = plt.subplot(gs3[0])
line6, = ax19.plot(time_data_sustr, sustr_avg, color='salmon', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_wind[:938] for July,
line61, = ax19.plot(time_data_svstr, svstr_avg, color='dodgerblue', label='$\\bf{Across-Shore}$', linewidth=8)
ax19.set_ylabel('Surface \nStress \n(N/m\u00b2)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax19.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax19.get_xticklabels(), visible=True)
xticks1 = ax19.xaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)


# Currents
ax20 = plt.subplot(gs3[1], sharex= ax19)
line7, = ax20.plot(time_data_cur[:-31], ubar_avg_masked[:-31]*100, color='orangered', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_cur[:313] for July 
line71, = ax20.plot(time_data_cur[:-31], vbar_avg_masked[:-31]*100, color='cornflowerblue', label='$\\bf{Across-Shore}$', linewidth=8)
ax20.set_ylabel('Larger-Scale \nCurrent \nSpeed \n(cm/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=118, va='center')
#ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
ax20.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax20.get_xticklabels(), visible=False)
yticks20 = ax20.yaxis.get_major_ticks()
#ax0.set_xlabel('Time', fontsize=30)

# Waves
# shared axis X
ax21 = plt.subplot(gs3[2], sharex = ax19)
line8, = ax21.plot(time_data_wave[:-117], swh_avg_masked[:-117], color='purple', label='$\\bf{Significant Wave Height}$', linewidth=8) #time_data_wave[:313] for July 
#ax2.axhspan(ax2.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
#ax2.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
plt.setp(ax21.get_xticklabels(), visible=False)
ax21.set_ylabel('Significant \nWave \nHeight \n(m)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=130, va='center')
# remove last tick label for the second subplot
yticks21 = ax21.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River water discharge
# shared axis X
ax22 = plt.subplot(gs3[3], sharex = ax19)
line9, = ax22.plot(time_data_riv[:-4], water_dis_tot[:-4], color='black', label='$\\bf{Total}$', linewidth=5) #time_data_riv[:40] for July 
# Colville
ax22.plot(time_data_riv[:-4], water_dis_col[:-4], label='$\\bf{Colville}$', linewidth=5, 
            color='#5EF38C')
# Kukpuk 
ax22.plot(time_data_riv[:-4], water_dis_kup[:-4], label='$\\bf{Kuparik}$', linewidth=5,
            color='#F43ECF')
plt.setp(ax22.get_xticklabels(), visible=False)
#ax3.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax22.set_ylabel('Water \nDischarge \n(m\u00b3/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# remove last tick label for the second subplot
yticks22 = ax22.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# River sediment discharge
# shared axis X
ax23 = plt.subplot(gs3[4], sharex = ax19)
line10, = ax23.plot(time_data_riv[:-4], water_sed_tot_kgs_2[:-4], color='black', label='$\\bf{Total}$', linewidth=5) #time_data_riv[:40] for July 
# Colville
ax23.plot(time_data_riv[:-4], water_sed_col_kgs_2[:-4], label='$\\bf{Colville}$', linewidth=5,
            color='#5EF38C')
# Kukpuk 
ax23.plot(time_data_riv[:-4], water_sed_kup_kgs_2[:-4], label='$\\bf{Kuparik}$', linewidth=5, 
            color='#F43ECF')
plt.setp(ax23.get_xticklabels(), visible=False)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax23.set_ylabel('River \nSediment \nLoad \n(kg/s)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=110, va='center')
# remove last tick label for the second subplot
yticks23 = ax23.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# Sea ice concentration 
# shared axis X
ax24 = plt.subplot(gs3[5], sharex = ax19)
line11, = ax24.plot(time_data_ice[:-4], mean_cover_masked[:-4], color='green', label='$\\bf{Sea Ice Concentration}$', linewidth=8) #time_data_riv[:40] for July 
plt.setp(ax24.get_xticklabels(), visible=True)
#ax4.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
ax24.set_ylabel('Sea Ice \nConc. \n(fraction \nof grid cell)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=115, va='center')
# remove last tick label for the second subplot
yticks24 = ax24.yaxis.get_major_ticks()
#yticks[-1].label1.set_visible(False)

# =============================================================================
# # Surface stress
# ax6 = plt.subplot(gs[6], sharex = ax0)
# line6, = ax6.plot(time_data_sustr, sustr_avg, color='salmon', label= '$\\bf{Along-Shore}$', linewidth=8) #time_data_wind[:938] for July,
# line61, = ax6.plot(time_data_svstr, svstr_avg, color='dodgerblue', label='$\\bf{Across-Shore}$', linewidth=8)
# ax6.set_ylabel('Surface \nStress \n(N/m\u00b2)', fontsize=40, fontweight='bold', rotation='horizontal', labelpad=120, va='center')
# #ax0.axhspan(ax0.get_ylim()[0], 0, facecolor='lemonchiffon', alpha=0.5)
# ax6.axhline(y=0.0, color='k', linestyle='--', linewidth=5)
# plt.setp(ax6.get_xticklabels(), visible=True)
# xticks1 = ax6.xaxis.get_major_ticks()
# #ax0.set_xlabel('Time', fontsize=30)
# =============================================================================

# put legend on first subplot
#ax0.legend((line0, line1), ('red line', 'blue line'))
ax19.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
ax20.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax2.legend(fontsize=40)
ax22.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
ax23.legend(fontsize=35, labelspacing=0.05, ncol=2, columnspacing=0.92)
#ax3.legend(fontsize=40)
#ax4.legend(fontsize=40)
#ax3.set_xlabel('Month in 2019')
#ax6.legend(fontsize=35, loc='lower left', labelspacing=0.05, ncol=2, columnspacing=0.92)
plt.xlabel('\nDate', fontsize=40, fontweight='bold')

# Add text labels for the panels (a, b, c, etc.)
# Use these if there are 6 subplots
plt.text(0.135, 0.849, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.727, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.595, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.47, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.345, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
plt.text(0.135, 0.22, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
# Use these if there are 7 subplots
#plt.text(0.135, 0.855, 'a)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.742, 'b)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.638, 'c)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.528, 'd)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.420, 'e)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.310, 'f)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)
#plt.text(0.135, 0.203, 'g)', fontsize=40, fontweight='bold', transform=plt.gcf().transFigure)

# Shade time periods of interest
# =============================================================================
# # Moccasin for periods looked at and plum for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-26', '2019-09-30', facecolor='plum', alpha=0.5) # cruise
# =============================================================================
# =============================================================================
# # Plum for periods looked at and moccasin for model-obs comp
# #ax13.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax13.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax13.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax13.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax13.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax14.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax14.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax14.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax14.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax14.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax15.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax15.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax15.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax15.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax15.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax16.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax16.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax16.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax16.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax16.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax17.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax17.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax17.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax17.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax17.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# #ax18.axvspan('2019-08-30', '2019-09-04', facecolor='moccasin', alpha=0.5)
# ax18.axvspan('2019-09-05', '2019-09-19', facecolor='plum', alpha=0.5)
# ax18.axvspan('2019-10-13', '2019-10-22', facecolor='plum', alpha=0.5)
# #ax18.axvspan('2019-09-26', '2019-09-30', facecolor='moccasin', alpha=0.5) # cruise Eidam
# #ax18.axvspan('2019-07-01', '2019-09-23', facecolor='moccasin', alpha=0.5) # CODA 2 and 3
# 
# =============================================================================

#fig1.set_tight_layout()

plt.tight_layout


# remove vertical gap between subplots
#plt.setp(ax0.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=.08)
#plt.show()




# --------------------------------------------------------------------------------
# -------------------------------------- Save Masks ------------------------------
# --------------------------------------------------------------------------------
# Save all of these masks to netcdf files so they can easily be read into 
# other scripts 
# Rho masks
# zeros and ones
# =============================================================================
# ds_mask_rho_zeros_ones = xr.DataArray({'nudge_mask_rho': (('eta_rho', 'xi_rho'), nudge_mask_rho)},
#                                      coords={'eta_rho': grid.eta_rho.values, 'xi_rho': grid.xi_rho.values},
#                                      dims=['eta_rho', 'xi_rho'])
# =============================================================================
ds_mask_rho_zeros_ones = xr.DataArray(data=nudge_mask_rho,
                                     dims=['eta_rho', 'xi_rho'],
                                     coords=dict(eta_rho=(['eta_rho'], grid.eta_rho.values), 
                                                 xi_rho=(['xi_rho'], grid.xi_rho.values)),
                                     name='nudge_mask_rho')
#ds_mask_rho_zeros_ones.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_rho_zeros_ones.nc')

# ones and nans
ds_mask_rho_ones_nans = xr.DataArray(data=nudge_mask_rho_nan,
                                     dims=['eta_rho', 'xi_rho'],
                                     coords=dict(eta_rho=(['eta_rho'], grid.eta_rho.values), 
                                                 xi_rho=(['xi_rho'], grid.xi_rho.values)),
                                     name='nudge_mask_rho_nan')
#ds_mask_rho_ones_nans.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_rho_ones_nans.nc')

# U masks 
# zeros and ones
ds_mask_u_zeros_ones = xr.DataArray(data=nudge_mask_u,
                                      dims=['eta_u', 'xi_u'],
                                     coords=dict(eta_u=(['eta_u'], grid.eta_u.values), 
                                                 xi_u=(['xi_u'], grid.xi_u.values)),
                                     name='nudge_mask_u')
#ds_mask_u_zeros_ones.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_u_zeros_ones.nc')

# ones and nans
ds_mask_u_ones_nans = xr.DataArray(data=nudge_mask_u_nan,
                                      dims=['eta_u', 'xi_u'],
                                     coords=dict(eta_u=(['eta_u'], grid.eta_u.values), 
                                                 xi_u=(['xi_u'], grid.xi_u.values)),
                                     name='nudge_mask_u_nan')
#ds_mask_u_ones_nans.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_u_ones_nans.nc')

# V masks 
# zeros and ones
ds_mask_v_zeros_ones = xr.DataArray(data=nudge_mask_v,
                                      dims=['eta_v', 'xi_v'],
                                     coords=dict(eta_v=(['eta_v'], grid.eta_v.values), 
                                                 xi_v=(['xi_v'], grid.xi_v.values)),
                                     name='nudge_mask_v')
#ds_mask_v_zeros_ones.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_v_zeros_ones.nc')

# ones and nans
ds_mask_v_ones_nans = xr.DataArray(data=nudge_mask_v_nan,
                                      dims=['eta_v', 'xi_v'],
                                     coords=dict(eta_v=(['eta_v'], grid.eta_v.values), 
                                                 xi_v=(['xi_v'], grid.xi_v.values)),
                                     name='nudge_mask_v_nan')
#ds_mask_v_ones_nans.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Code/Nudge_masks/nudge_mask_v_ones_nans.nc')


# -------------------------------------------------------------------------------
# ---- Make a netcdf to hold the ubar and vbar clm data used for plotting 
# -------------------------------------------------------------------------------
# Set up the data
depth_avg_currents_clm = xr.Dataset(
    data_vars=dict(
        ubar_avg_masked=(['ocean_time'], ubar_avg_masked),
        vbar_avg_masked=(['ocean_time'], vbar_avg_masked),
        ),
    coords=dict(
        ocean_time=time_data_cur
        ),
    attrs=dict(units='meter per second', description='depth-averaged water momentum, averaged over the domain'))

# Save this to a netcdf
#depth_avg_currents_clm.to_netcdf('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/Paper1/Data/fig3_ubar_vbar_clm.nc')


