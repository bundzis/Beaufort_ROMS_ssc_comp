################## Calculate Riverine Sediment Masses ################
# The purpose of this script is to calculate and plot time series of 
# where riverine sediments are over time in the model relative to
# the amount that has entered from each river.
#
# Notes:
# - This works best in xroms
# - This uses the samme code as plot_end_comp_river_muds.ipynb but runs
#   as batch script for more mem
########################################################################


# Load in the packages 
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cmocean
from shapely.geometry import Polygon
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import os
import matplotlib.colors as mcolors
from matplotlib import colors
from matplotlib import cm, ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator, LogFormatterExponent, LogFormatterSciNotation, LogFormatter)
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.ticker as tick
from matplotlib import cm, ticker
import warnings
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterExponent
crs = ccrs.PlateCarree()
warnings.filterwarnings("ignore") #turns off annoying warnings
#Cartopy
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                edgecolor='face',
                                facecolor=cfeature.COLORS['land'])


# -------- General Things ----------
# set a universal fontsize 
fontsize = 20 # 32 #20

# Set the tick size for all plots
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize)

# Prevent tick labels from overlapping
matplotlib.rcParams['xtick.major.pad'] = 12 # 28 # 12
matplotlib.rcParams['ytick.major.pad'] = 12 # 18 # 12


# -------- Load Data ----------
# Load in the model grid
#grid = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Grids/KakAKgrd_shelf_big010_smooth006.nc') 
grid = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Include/KakAKgrd_shelf_big010_smooth006.nc')

# Pull out some dimensions
eta_rho_len = len(grid.eta_rho)
xi_rho_len = len(grid.xi_rho)
s_rho_len = int(20)

# Load in all of the model output
# Unaggregated
model_output_unag = xr.open_mfdataset('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Unaggregated/dbsed0001_10rivs_unaggregated/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc')
# Aggregated 
model_output_agg = xr.open_mfdataset('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Aggregated/dbsed0001_10rivs/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')


# Load in the river forcing file 
river_frc_std = xr.open_dataset('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Include/river_forcing_file_beaufort_shelf_10rivs_13seabed_blaskey_data_sagDSS3_rating_001.nc')



# -------- Define Functions ----------
# Make a function that takes the river input, the idnices, and returns the discharge and 
# sediemnt load for a given river
def get_river_dis_sed(river_frc, first_idx, run, seds, last_idx=2000):
    """
    The purpose of this function is to get atime series fo the river discharge
    and sediment load for a given river.
    
    Inputs:
        - river_frc: River forcing file loaded via xarray
        - first_idx: First index of the river in the forcing file (or only index if only one)
        - last_idx: Last index of the river in the forcing file
        - run: string of the run, either 'std' or 'double'
        - seds: string of if you want 'all' river sediments (agg and unag together) or 'unique' (the unique sediments)
    Outputs:
        - river_dis: Time series of river discharge for that river
        - river_sed: Tiem series of river sediment load for that river
    """
    print('first_idx: ', first_idx)
    print('last_idx: ', last_idx)
    
    # Get the transport
    # Check if the river covers multiple cells
    if (last_idx != 2000):
        # Get the discharge 
        river_dis = river_frc.river_transport[:,first_idx:last_idx+1].sum(dim='river')
        #print('river_dis shape multiple cells: ', np.shape(river_dis))
    else:
        river_dis = river_frc.river_transport[:,first_idx]
        #print('river_dis shape one cells: ', np.shape(river_dis))
    
    # Get the sediment load
    # Add together all of the sediment classes
    if run == 'std':
        if seds == 'all':
            # Unaggregated and Aggregated together 
            riv_all_ssc = (river_frc.river_sand_01 + river_frc.river_sand_02 + river_frc.river_sand_03 + 
                        river_frc.river_mud_01 + river_frc.river_mud_02 + river_frc.river_mud_03 + river_frc.river_mud_04 + 
                        river_frc.river_mud_05 + river_frc.river_mud_06 + river_frc.river_mud_07 + river_frc.river_mud_08 + 
                        river_frc.river_mud_09 + river_frc.river_mud_10 + river_frc.river_mud_11 + river_frc.river_mud_12 + 
                        river_frc.river_mud_13 + river_frc.river_mud_14 + river_frc.river_mud_15 + river_frc.river_mud_16 + 
                        river_frc.river_mud_17 + river_frc.river_mud_18 + river_frc.river_mud_19 + river_frc.river_mud_20 + 
                        river_frc.river_mud_21 + river_frc.river_mud_22 + river_frc.river_mud_23 + river_frc.river_mud_24) 
        if seds == 'unique':
            # Just the unique classes
            print(river_frc.river_mud_15[80,0].values)
            riv_all_ssc = (river_frc.river_mud_15 + river_frc.river_mud_16 + 
                        river_frc.river_mud_17 + river_frc.river_mud_18 + river_frc.river_mud_19 + river_frc.river_mud_20 + 
                        river_frc.river_mud_21 + river_frc.river_mud_22 + river_frc.river_mud_23 + river_frc.river_mud_24)
            # check if this is adding the way we expect and want
            print(riv_all_ssc[80,0].values)
    if run == 'double':
        riv_all_ssc = (river_frc.river_sand_01 + river_frc.river_sand_02 + river_frc.river_sand_03 + 
                      river_frc.river_mud_01 + river_frc.river_mud_02 + river_frc.river_mud_03 + river_frc.river_mud_04)
        
    
    # Multiply to get the load and save ssc
    if last_idx != 2000:
        river_sed = (river_frc.river_transport[:,first_idx:last_idx+1]*riv_all_ssc[:,first_idx:last_idx+1]).sum(dim='river') # kg/s
        river_ssc = riv_all_ssc[:,first_idx:last_idx+1].sum(dim='river') # kg/m3
    else:
        river_sed = (river_frc.river_transport[:,first_idx]*riv_all_ssc[:,first_idx]) # kg/s
        river_ssc = riv_all_ssc[:,first_idx] # kg/m3
        
    # Return these values
    return(river_dis, river_sed, river_ssc)

# Make a function to get fractions and print 
def calculate_print_percent_riv_seabed_water(river_name, river_tot, river_bed, river_water):
    """
    This function takes the river name, total river sediment at certain time, 
    mass in bed, and mass in water at that same time and prints out percentages and
    totals.

    Inputs:
    - river_name: string of river name, 'Kalikpik'
    - river_tot: total mass of sediment from that river 
                    received at same time as other inputs (kg)
    - river_bed: total mass of river sed in the seabed at certain time (kg)
    - river_water: total mass of river sed in the water at certain time (kg)
    """

    # Get the ratios

    # Print stuff
    print(river_name)
    print('Total Unique kg Delivered: ', river_tot.values)
    print('Total Unique kg Bed (*broken*): ', river_bed.values)
    print('Total Unique kg Water: ', river_water.values)
    print('Percent in Seabed (*broken*): ', (river_bed/river_tot).values*100)
    print('Percent in Water: ', (river_water/river_tot).values*100)
    print('Percent Unaccounted for (*broken*): ', (100-(((river_bed/river_tot).values*100) + ((river_water/river_tot).values*100))))



# -------- Prep Riverine Input ----------
# Get discharge, sediment load, and SSC for each river
# Call the function for each river 
# Standard run 
water_dis_tot_std, water_sed_tot_std, water_ssc_tot_std = get_river_dis_sed(river_frc_std, 0, 'std', 'unique', -2)
water_dis_kal_std, water_sed_kal_std, water_ssc_kal_std = get_river_dis_sed(river_frc_std, 0, 'std', 'unique')
water_dis_col_std, water_sed_col_std, water_ssc_col_std = get_river_dis_sed(river_frc_std, 1, 'std', 'unique', 6)
water_dis_sag_std, water_sed_sag_std, water_ssc_sag_std = get_river_dis_sed(river_frc_std, 7, 'std', 'unique', 9)
water_dis_fis_std, water_sed_fis_std, water_ssc_fis_std = get_river_dis_sed(river_frc_std, 10, 'std', 'unique')
water_dis_kup_std, water_sed_kup_std, water_ssc_kup_std = get_river_dis_sed(river_frc_std, 11, 'std', 'unique')
water_dis_sta_std, water_sed_sta_std, water_ssc_sta_std = get_river_dis_sed(river_frc_std, 12, 'std', 'unique')
water_dis_can_std, water_sed_can_std, water_ssc_can_std = get_river_dis_sed(river_frc_std, 13, 'std', 'unique')
water_dis_kat_std, water_sed_kat_std, water_ssc_kat_std = get_river_dis_sed(river_frc_std, 14, 'std', 'unique')
water_dis_hul_std, water_sed_hul_std, water_ssc_hul_std = get_river_dis_sed(river_frc_std, 15, 'std', 'unique')
water_dis_jag_std, water_sed_jag_std, water_ssc_jag_std = get_river_dis_sed(river_frc_std, 16, 'std', 'unique')

# Now we have the kg/s of sediment from each river but want the total
# Set delta time for model output
delta_time = 86400 # 86400 seconds = 1 day (frequency of river inputs); 14400 seconds = 4 hours 

# Set last time index
last_time_idx = -4

# Multiply the river sediment load (kg/s) by s to get total kgs
print(np.shape(water_sed_kal_std))
kal_sed_kg = water_sed_kal_std * delta_time # kg
print(np.shape(kal_sed_kg))
col_sed_kg = water_sed_col_std * delta_time
sag_sed_kg = water_sed_sag_std * delta_time
fis_sed_kg = water_sed_fis_std * delta_time
kup_sed_kg = water_sed_kup_std * delta_time
sta_sed_kg = water_sed_sta_std * delta_time
can_sed_kg = water_sed_can_std * delta_time
kat_sed_kg = water_sed_kat_std * delta_time
hul_sed_kg = water_sed_hul_std * delta_time
jag_sed_kg = water_sed_jag_std * delta_time

# Multiply each river sed by this delta time
kal_tot_sed_kg = np.cumsum(kal_sed_kg[:last_time_idx])
col_tot_sed_kg = np.cumsum(col_sed_kg[:last_time_idx])
sag_tot_sed_kg = np.cumsum(sag_sed_kg[:last_time_idx])
fis_tot_sed_kg = np.cumsum(fis_sed_kg[:last_time_idx])
kup_tot_sed_kg = np.cumsum(kup_sed_kg[:last_time_idx])
sta_tot_sed_kg = np.cumsum(sta_sed_kg[:last_time_idx])
can_tot_sed_kg = np.cumsum(can_sed_kg[:last_time_idx])
kat_tot_sed_kg = np.cumsum(kat_sed_kg[:last_time_idx])
hul_tot_sed_kg = np.cumsum(hul_sed_kg[:last_time_idx])
jag_tot_sed_kg = np.cumsum(jag_sed_kg[:last_time_idx])


# -------- Prep Model Output ----------
# Get the mass in the seabed and water column over time for certain river
# First set the horizontal dimensions
dx = 750 # m
dy = 600 # m
dz_full_unag = abs(model_output_unag.z_w[:,:-1,:,:].values - model_output_unag.z_w[:,1:,:,:].values) # m
dz_full_agg = abs(model_output_agg.z_w[:,:-1,:,:].values - model_output_agg.z_w[:,1:,:,:].values) # m

# Time series of mass in bed and Time series of mass in water column 
# Do all the other rivers, too
print('Starting unag calculations', flush=True)
# # -- Unaggregated --
# # Kalikpik
# kal_tot_sed_seabed_unag_full = (model_output_unag.mudmass_15[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# kal_tot_sed_water_unag_full = (model_output_unag.mud_15[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Colville
# col_tot_sed_seabed_unag_full = (model_output_unag.mudmass_16[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# col_tot_sed_water_unag_full = (model_output_unag.mud_16[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Sagavanirktok
# sag_tot_sed_seabed_unag_full = (model_output_unag.mudmass_17[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# sag_tot_sed_water_unag_full = (model_output_unag.mud_17[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Fish Creek
# fis_tot_sed_seabed_unag_full = (model_output_unag.mudmass_18[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# fis_tot_sed_water_unag_full = (model_output_unag.mud_18[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Kuparuk
# kup_tot_sed_seabed_unag_full = (model_output_unag.mudmass_19[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# kup_tot_sed_water_unag_full = (model_output_unag.mud_19[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Staines
# sta_tot_sed_seabed_unag_full = (model_output_unag.mudmass_20[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# sta_tot_sed_water_unag_full = (model_output_unag.mud_20[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Canning 
# can_tot_sed_seabed_unag_full = (model_output_unag.mudmass_21[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# can_tot_sed_water_unag_full = (model_output_unag.mud_21[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Katakturuk
# kat_tot_sed_seabed_unag_full = (model_output_unag.mudmass_22[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# kat_tot_sed_water_unag_full = (model_output_unag.mud_22[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Hulahula
# hul_tot_sed_seabed_unag_full = (model_output_unag.mudmass_23[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# hul_tot_sed_water_unag_full = (model_output_unag.mud_23[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# # Jago
# jag_tot_sed_seabed_unag_full = (model_output_unag.mudmass_24[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
# jag_tot_sed_water_unag_full = (model_output_unag.mud_24[:,:,:,:]*dx*dy*dz_full_unag).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()

print('Starting agg calculations', flush=True)
# -- Aggregated --
# Kalikpik
kal_tot_sed_seabed_agg_full = (model_output_agg.mudmass_15[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
kal_tot_sed_water_agg_full = (model_output_agg.mud_15[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Colville
col_tot_sed_seabed_agg_full = (model_output_agg.mudmass_16[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
col_tot_sed_water_agg_full = (model_output_agg.mud_16[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Sagavanirktok
sag_tot_sed_seabed_agg_full = (model_output_agg.mudmass_17[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
sag_tot_sed_water_agg_full = (model_output_agg.mud_17[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Fish Creek
fis_tot_sed_seabed_agg_full = (model_output_agg.mudmass_18[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
fis_tot_sed_water_agg_full = (model_output_agg.mud_18[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Kuparuk
kup_tot_sed_seabed_agg_full = (model_output_agg.mudmass_19[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
kup_tot_sed_water_agg_full = (model_output_agg.mud_19[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Staines
sta_tot_sed_seabed_agg_full = (model_output_agg.mudmass_20[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
sta_tot_sed_water_agg_full = (model_output_agg.mud_20[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Canning 
can_tot_sed_seabed_agg_full = (model_output_agg.mudmass_21[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
can_tot_sed_water_agg_full = (model_output_agg.mud_21[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Katakturuk
kat_tot_sed_seabed_agg_full = (model_output_agg.mudmass_22[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
kat_tot_sed_water_agg_full = (model_output_agg.mud_22[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Hulahula
hul_tot_sed_seabed_agg_full = (model_output_agg.mudmass_23[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
hul_tot_sed_water_agg_full = (model_output_agg.mud_23[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()
# Jago
jag_tot_sed_seabed_agg_full = (model_output_agg.mudmass_24[:,:,:,:]*dx*dy).sum(dim=['Nbed', 'eta_rho', 'xi_rho']).compute()
jag_tot_sed_water_agg_full = (model_output_agg.mud_24[:,:,:,:]*dx*dy*dz_full_agg).sum(dim=['s_rho', 'eta_rho', 'xi_rho']).compute()




# -------- Plot Time Series ----------
# Plot these sums 

# fig, ax = plt.subplots(figsize=(12,4))

# ax.plot(model_output_unag.ocean_time.values, col_tot_sed_seabed_unag_full, color='brown', label='seabed')
# ax.plot(model_output_unag.ocean_time.values, col_tot_sed_water_unag_full, color='blue', label='water')
# ax.plot(model_output_unag.ocean_time.values, (col_tot_sed_seabed_unag_full + col_tot_sed_water_unag_full), color='black', label='total (bed+water)')
# ax.plot(river_frc_std.river_time[:121].values, col_tot_sed_kg, color='green', label='River cumsum')

# ax.legend()
# ax.set_title('Colville')
# ax.set_xlabel('Time')
# ax.set_ylabel('Mass (kg)')

# #plt.savefig('/projects/brun1463/ROMS/colville_unag_mud_bed_water_river_001.png')

print('Saving data', flush=True)
# --------- Save the Data ----------
# Unaggregated
# # Make an xarray dataset to save the data to 
# river_sed_data_by_river_unag = xr.Dataset(
#     data_vars=dict(
#         kal_tot_sed_seabed_unag_full=(['ocean_time'], kal_tot_sed_seabed_unag_full.values),
#         kal_tot_sed_water_unag_full=(['ocean_time'], kal_tot_sed_water_unag_full.values),
#         col_tot_sed_seabed_unag_full=(['ocean_time'], col_tot_sed_seabed_unag_full.values),
#         col_tot_sed_water_unag_full=(['ocean_time'], col_tot_sed_water_unag_full.values),
#         sag_tot_sed_seabed_unag_full=(['ocean_time'], sag_tot_sed_seabed_unag_full.values),
#         sag_tot_sed_water_unag_full=(['ocean_time'], sag_tot_sed_water_unag_full.values),
#         fis_tot_sed_seabed_unag_full=(['ocean_time'], fis_tot_sed_seabed_unag_full.values),
#         fis_tot_sed_water_unag_full=(['ocean_time'], fis_tot_sed_water_unag_full.values),
#         kup_tot_sed_seabed_unag_full=(['ocean_time'], kup_tot_sed_seabed_unag_full.values),
#         kup_tot_sed_water_unag_full=(['ocean_time'], kup_tot_sed_water_unag_full.values),
#         sta_tot_sed_seabed_unag_full=(['ocean_time'], sta_tot_sed_seabed_unag_full.values),
#         sta_tot_sed_water_unag_full=(['ocean_time'], sta_tot_sed_water_unag_full.values),
#         can_tot_sed_seabed_unag_full=(['ocean_time'], can_tot_sed_seabed_unag_full.values),
#         can_tot_sed_water_unag_full=(['ocean_time'], can_tot_sed_water_unag_full.values),
#         kat_tot_sed_seabed_unag_full=(['ocean_time'], kat_tot_sed_seabed_unag_full.values),
#         kat_tot_sed_water_unag_full=(['ocean_time'], kat_tot_sed_water_unag_full.values),
#         hul_tot_sed_seabed_unag_full=(['ocean_time'], hul_tot_sed_seabed_unag_full.values),
#         hul_tot_sed_water_unag_full=(['ocean_time'], hul_tot_sed_water_unag_full.values),
#         jag_tot_sed_seabed_unag_full=(['ocean_time'], jag_tot_sed_seabed_unag_full.values),
#         jag_tot_sed_water_unag_full=(['ocean_time'], jag_tot_sed_water_unag_full.values)
#         ),
#     coords=dict(
#         ocean_time=('ocean_time', model_output_unag.ocean_time.values)
#         ),
#     attrs=dict(description='Time series of riverine unaggregated muds in the water column and seabed from Unique Unaggregated Simulation'))

# Save this to a netcdf
#river_sed_data_by_river_unag.to_netcdf('/projects/brun1463/ROMS/river_sed_seabed_water_timeseries_unag.nc')

# Make another one for aggregated and save that 
# Make an xarray dataset to save the data to 
river_sed_data_by_river_agg = xr.Dataset(
    data_vars=dict(
        kal_tot_sed_seabed_agg_full=(['ocean_time'], kal_tot_sed_seabed_agg_full.values),
        kal_tot_sed_water_agg_full=(['ocean_time'], kal_tot_sed_water_agg_full.values),
        col_tot_sed_seabed_agg_full=(['ocean_time'], col_tot_sed_seabed_agg_full.values),
        col_tot_sed_water_agg_full=(['ocean_time'], col_tot_sed_water_agg_full.values),
        sag_tot_sed_seabed_agg_full=(['ocean_time'], sag_tot_sed_seabed_agg_full.values),
        sag_tot_sed_water_agg_full=(['ocean_time'], sag_tot_sed_water_agg_full.values),
        fis_tot_sed_seabed_agg_full=(['ocean_time'], fis_tot_sed_seabed_agg_full.values),
        fis_tot_sed_water_agg_full=(['ocean_time'], fis_tot_sed_water_agg_full.values),
        kup_tot_sed_seabed_agg_full=(['ocean_time'], kup_tot_sed_seabed_agg_full.values),
        kup_tot_sed_water_agg_full=(['ocean_time'], kup_tot_sed_water_agg_full.values),
        sta_tot_sed_seabed_agg_full=(['ocean_time'], sta_tot_sed_seabed_agg_full.values),
        sta_tot_sed_water_agg_full=(['ocean_time'], sta_tot_sed_water_agg_full.values),
        can_tot_sed_seabed_agg_full=(['ocean_time'], can_tot_sed_seabed_agg_full.values),
        can_tot_sed_water_agg_full=(['ocean_time'], can_tot_sed_water_agg_full.values),
        kat_tot_sed_seabed_agg_full=(['ocean_time'], kat_tot_sed_seabed_agg_full.values),
        kat_tot_sed_water_agg_full=(['ocean_time'], kat_tot_sed_water_agg_full.values),
        hul_tot_sed_seabed_agg_full=(['ocean_time'], hul_tot_sed_seabed_agg_full.values),
        hul_tot_sed_water_agg_full=(['ocean_time'], hul_tot_sed_water_agg_full.values),
        jag_tot_sed_seabed_agg_full=(['ocean_time'], jag_tot_sed_seabed_agg_full.values),
        jag_tot_sed_water_agg_full=(['ocean_time'], jag_tot_sed_water_agg_full.values)
        ),
    coords=dict(
        ocean_time=('ocean_time', model_output_agg.ocean_time.values)
        ),
    attrs=dict(description='Time series of riverine aggregated muds in the water column and seabed from Unique Aggregated Simulation'))

# Save this to a netcdf
river_sed_data_by_river_agg.to_netcdf('/projects/brun1463/ROMS/river_sed_seabed_water_timeseries_agg.nc')

