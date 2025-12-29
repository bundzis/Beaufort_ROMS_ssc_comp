################ Plot kg Sediment In Regions as Time Series ######################
# The purpose of this script is to make a plot showing the different kg of 
# sediment in different regions of the shelf over time. The vision right now
# is to have one plot per sediment type, preferrably for the different rivers
# meaning there would be 13 plots total
#
# Notes:
# - Right now, this only considers SSC so sediment in the water column and 
#   excludes sediment in the seabed, though that could be good to plot, too
#   - So consider adding in version that looks at amount in the seabed 
# - Start with just rivers but consider doing same for sediment from 
#   different seabed sections 
# - This has been updated to look at 2020 version that only has 10 rivers 
##################################################################################


# Load in the packages 
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import transforms 
from matplotlib import cm, ticker
from glob import glob
import cmocean
#import matplotlib.ticker as tick
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# Set a universal fontsize
fontsize = 25

# Set the tick size for all plots
matplotlib.rc('xtick', labelsize=fontsize) 
matplotlib.rc('ytick', labelsize=fontsize)

# Prevent tick labels from overlapping
matplotlib.rcParams['xtick.major.pad'] = 12
matplotlib.rcParams['ytick.major.pad'] = 12

# Load in the grid
grid = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Include/KakAKgrd_shelf_big010_smooth006.nc')
#grid = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Grids/KakAKgrd_shelf_big010_smooth006.nc') # UPDATE PATH


# Pull out some dimensions
eta_rho_len = len(grid.eta_rho)
xi_rho_len = len(grid.xi_rho)
s_rho_len = int(20)


# Load in the rho masks 
mask_rho_nan = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_ones_nans.nc') # UPDATE PATH
mask_rho_zeros = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_zeros_ones.nc')
#mask_rho_nan = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_ones_nans.nc')
#mask_rho_zeros = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_zeros_ones.nc')


# Load in the river forcing file 
#river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/Rivers/river_forcing_file_beaufort_shelf_13rivs_13seabed_radr_data_003.nc')
#river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/Rivers/river_forcing_file_beaufort_shelf_13rivs_13seabed_radr_data_003.nc') # Same in unaggregated and aggregated 
# (2020)
river_frc = xr.open_dataset('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Include/river_forcing_file_beaufort_shelf_10rivs_13seabed_blaskey_data_sagDSS3_rating_001.nc')


# =============================================================================
# # Make a bunch of functions 
# # Make a function to pull out the time series of a given sediment class
# def get_ssc_timeseries_by_class(filename, sediment_class):
#     """
#     This function goes through a given model output file and pulls
#     out the spatial time series of SSC for a given sediment class.
# 
#     Parameters
#     ----------
#     filename : The name/path of the model output file
#     sediment_class : The desired sediment class
# 
#     Returns
#     -------
#     None.
# 
#     """
#     # Load in the model output
#     model_output = xr.open_dataset(filename)
# =============================================================================
    
    
# Make a function to get the depth-integrated kg sediment per cell in space
# as a time series
def get_depth_int_ssc_timeseries_by_class(filename, sediment_class):
    """
    This function takes a model output file, pulls out the desired 
    sediment class, gets the depth-integrated SSC as a time series.

    Parameters
    ----------
    filename : The name/path of the model output file
    sediment_class : The desired sediment class (str)

    Returns
    -------
    None.

    """
    
    # Load in the model output
    model_output = xr.open_dataset(filename)
    
    # Pull out the desired sediment class
    ssc_1sed = model_output[sediment_class]
    
    # To collapse to horizontal, multiply each layer by its
    # thickness
    # Calculate the time-varying thickness of the cells
    dz = abs(model_output.z_w[:,:-1,:,:].values - model_output.z_w[:,1:,:,:].values)
    
    # Multiply the SSC by thick thickness
    depth_int_ssc_1sed_tmp = (ssc_1sed*dz)
    
    # Then depth-integrated by summing over depth and dividing by dy
    depth_int_ssc_1sed = (depth_int_ssc_1sed_tmp.sum(dim='s_rho'))
    
    # Now we have SSC in kg/m2 for each grid cell in horizontal
    # Return this value - can multiply by dx and dy outside the function
    return(depth_int_ssc_1sed)
    

# Define a function to pull out the length of time in the model run
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




# Loop through model output and call the function
# First, get all the file names 
# -- Aggregated --
#file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Kakak3_Alpine_scratch/ocean_his_biggrid010_gridwindsiniwaves_rivs_si_smooth006_nobulk_chaflaradnudclm_dbsed0007_*.nc')
#file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Kakak3_Alpine_scratch/ocean_his_biggrid010_gridwindsiniwaves_rivs_si_smooth006_nobulk_chaflaradnudclm_dbsed0007_*.nc')
# dbsed0007
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/model_output/Full_run_0003_sponge_swell/ocean_his_biggrid010_gridwindsiniwaves_rivs_si_smooth006_nobulk_chaflaradnudclm_dbsed0007_*.nc')
# dbsed0008
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska/model_output/Full_run_0004_sponge_swell_1mms/ocean_his_biggrid010_gridwindsiniwaves_rivs_si_smooth006_nobulk_chaflaradnudclm_dbsed0008_*.nc')
# dbsed0008
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Outputs/dbsed0003/ocean_his_beaufort_rivers_14rivs_13seabed_dbsed0003_*.nc')
# Seabed sections
# dbsed0006
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Outputs/dbsed0006/ocean_his_beaufort_rivers_13rivs_13seabed_dbsed0006_*.nc')
# dbsed0009
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Outputs/dbsed0009/ocean_his_beaufort_rivers_13rivs_13seabed_dbsed0009_*.nc')
# 2020 dbsed0001 - full run 
#file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_002_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
# -- Unaggregated --
# dbsed0002
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_003/Model_Output/dbsed0002/ocean_his_beaufort_rivers_13rivs_13seabed_unaggregated_dbsed0002_*.nc')
# 2020 dbsed0001 - full run
file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_003_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc') 

# Sort them to be in order
file_names2 = sorted(file_names)

# Check to see if this worked
print(file_names2[0], flush=True)
print(file_names2[1], flush=True)
print(file_names2[2], flush=True)
print(file_names2[-1], flush=True)

# Pull out the number of files
num_files = len(file_names2)

# Pull out the length of time of the full run, the time steps, 
# and the length of time of each output file
full_time_len, time_steps, time_lengths = get_model_time(file_names2, num_files)


# Make some arrays to hold output
# One for each sediment type in the river muds - look at guide
depth_int_ssc_mud15 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud16 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud17 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud18 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud19 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud20 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud21 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud22 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud23 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
depth_int_ssc_mud24 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
#depth_int_ssc_mud25 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
#depth_int_ssc_mud26 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
#depth_int_ssc_mud27 = np.empty((full_time_len, eta_rho_len, xi_rho_len))
#depth_int_ssc_mud28 = np.empty((full_time_len, eta_rho_len, xi_rho_len))

# Set a time step to track which time step the loop is on
time_step = 0

# Loop through the model output
for j in range(num_files):
#for j in range(1):

    print('j: ', j, flush=True)
    
    # Call the function to process the output and Save these to the arrays 
    #print('time_step: ', time_step)
    #print('time_step + time_lengths[j]: ', time_step+time_lengths[j])
    start = int(time_step)
    end = int(time_step+time_lengths[j])
    
    # Mud15
    depth_int_ssc_mud15[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_15')
    # Mud16
    depth_int_ssc_mud16[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_16')
    # Mud17
    depth_int_ssc_mud17[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_17')
    # Mud18
    depth_int_ssc_mud18[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_18')
    # Mud19
    depth_int_ssc_mud19[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_19')
    # Mud20
    depth_int_ssc_mud20[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_20')
    # Mud21
    depth_int_ssc_mud21[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_21')
    # Mud22
    depth_int_ssc_mud22[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_22')
    # Mud23
    depth_int_ssc_mud23[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_23')
    # Mud24
    depth_int_ssc_mud24[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_24')
    # Mud25
    #depth_int_ssc_mud25[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_25')
    # Mud26
    #depth_int_ssc_mud26[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_26')
    # Mud27
    #depth_int_ssc_mud27[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_27')
    # Mud28
    #depth_int_ssc_mud28[start:end,:,:] = get_depth_int_ssc_timeseries_by_class(file_names2[j], 'mud_28')
    
    # Update the base time_step
    time_step = time_step + time_lengths[j]
    

# Okay now multiply by dx and dy, and rename to be by river and combine where needed
dx = 750 # meters
dy = 600 # meters 

# Kalikpik
kg_sed_mud15_kal = depth_int_ssc_mud15*dx*dy # kg
# Covlille
kg_sed_mud16_col = depth_int_ssc_mud16*dx*dy # kg
# Sagavanirktok
kg_sed_mud17_sag = depth_int_ssc_mud17*dx*dy # kg
# Fish Creek
kg_sed_mud18_fis = depth_int_ssc_mud18*dx*dy # kg
# Sakonowyak
#kg_sed_mud19_sak = depth_int_ssc_mud19*dx*dy # kg
# Kuparik
kg_sed_mud19_kup = (depth_int_ssc_mud19)*dx*dy # kg
# Putuligayuk
#kg_sed_mud21_put = depth_int_ssc_mud21*dx*dy # kg
# Staines
kg_sed_mud20_sta = depth_int_ssc_mud20*dx*dy # kg
# Canning
kg_sed_mud21_can = depth_int_ssc_mud21*dx*dy # kg
# Katakturuk
kg_sed_mud22_kat = depth_int_ssc_mud22*dx*dy # kg
# Hulahula
kg_sed_mud23_hul = depth_int_ssc_mud23*dx*dy # kg
# Jago
kg_sed_mud24_jag = depth_int_ssc_mud24*dx*dy # kg
# Siksik
#kg_sed_mud27_sik = depth_int_ssc_mud27*dx*dy # kg

# Check the shapes 
print(np.shape(kg_sed_mud24_jag))

# Multiply by mask to get region we trust 
# Set the number of cells in the sponge on each open boundary
c_west = 36
c_north = 45
c_east = 36
# Make it so land will appear
temp_mask = grid.mask_rho.copy()
temp_mask = np.where(temp_mask==0, np.nan, temp_mask)
# Mask, trim
# Make empty arrays to hold masked and trimmed versions 
kg_sed_mud15_kal_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud16_col_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud17_sag_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud18_fis_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud19_kup_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud20_sta_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud21_can_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud22_kat_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud23_hul_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_mud24_jag_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
#kg_sed_mud25_hul_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
#kg_sed_mud26_jag_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))
#kg_sed_mud27_sik_masked = np.empty((full_time_len,eta_rho_len,xi_rho_len))

# Loop through time
for t in range(full_time_len):
    print('t: ', t, flush=True)
    
    # Mask
    kg_sed_mud15_kal_masked[t,:,:] = kg_sed_mud15_kal[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud16_col_masked[t,:,:] = kg_sed_mud16_col[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud17_sag_masked[t,:,:] = kg_sed_mud17_sag[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud18_fis_masked[t,:,:] = kg_sed_mud18_fis[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    #kg_sed_mud19_sak_masked[t,:,:] = kg_sed_mud19_sak[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud19_kup_masked[t,:,:] = kg_sed_mud19_kup[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    #kg_sed_mud21_put_masked[t,:,:] = kg_sed_mud21_put[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud20_sta_masked[t,:,:] = kg_sed_mud20_sta[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud21_can_masked[t,:,:] = kg_sed_mud21_can[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud22_kat_masked[t,:,:] = kg_sed_mud22_kat[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud23_hul_masked[t,:,:] = kg_sed_mud23_hul[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    kg_sed_mud24_jag_masked[t,:,:] = kg_sed_mud24_jag[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
    #kg_sed_mud27_sik_masked[t,:,:] = kg_sed_mud27_sik[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan



# Multuply it by masks to get 0 - 10 m depth, 10  - 20 m depth, 20 - 30, 30 - 60
# Make masks for these depths 
# Make a function to mask the data but that takes two thresholds
def masked_array_lowhigh_2dloop(data, lower, upper):
    """
    This function takes an array and masks all values that are less
    than a certain given threshold. The functions returns 1 for areas that meet 
    the condition and 0 for areas that don't. So areas where the array is less
    than the threshold get returned as 1 and areas greater than the threshold
    are returned as 0. This function maintains the shape of the array.
    
    """
    mask_tmp = np.empty_like((data))
    
    # Loop through dimension 1
    for i in range(len(data[:,0])):
        # Loop through dimension 2
        for j in range(len(data[0,:])):
            # Compare against threshold 
            value = data[i,j]
            if lower < value <= upper:
                mask_tmp[i,j] = 1
            else:
                mask_tmp[i,j] = 0
    
    
    return (mask_tmp).astype(int)
 
# Call the function to make the mask
# First make an array of the bathymetry of the grid pre-masked
h_masked = grid.h * mask_rho_nan.nudge_mask_rho_nan
# 0 - 10 m
h_masked1 = h_masked.copy()
inner_10m_mask_rho = masked_array_lowhigh_2dloop(h_masked1, 2, 10)
# 10 - 20 m
h_masked2 = h_masked.copy()
outer_10_20m_mask_rho = masked_array_lowhigh_2dloop(h_masked2, 10, 20)
# 20 - 30 m 
h_masked3 = h_masked.copy()
outer_20_30m_mask_rho = masked_array_lowhigh_2dloop(h_masked3, 20, 30)
# 30 - 60 m depth
h_masked4 = h_masked.copy()
outer_30_60m_mask_rho = masked_array_lowhigh_2dloop(h_masked4, 30, 60)

# Make the masks nan where they are 0 so that these out of bounds areas are 
# nanned out 
# 0 - 10 m
inner_10m_mask_rho_nan_idx = np.where(inner_10m_mask_rho == 0.0)
inner_10m_mask_rho_nan = inner_10m_mask_rho .copy()
inner_10m_mask_rho_nan = inner_10m_mask_rho_nan.astype('float')
inner_10m_mask_rho_nan[inner_10m_mask_rho_nan_idx] = np.nan
# 10 - 20 m
outer_10_20m_mask_rho_nan_idx = np.where(outer_10_20m_mask_rho == 0.0)
outer_10_20m_mask_rho_nan = outer_10_20m_mask_rho.copy()
outer_10_20m_mask_rho_nan = outer_10_20m_mask_rho_nan.astype('float')
outer_10_20m_mask_rho_nan[outer_10_20m_mask_rho_nan_idx] = np.nan
# 20 - 30 m
outer_20_30m_mask_rho_nan_idx = np.where(outer_20_30m_mask_rho == 0.0)
outer_20_30m_mask_rho_nan = outer_20_30m_mask_rho.copy()
outer_20_30m_mask_rho_nan = outer_20_30m_mask_rho_nan.astype('float')
outer_20_30m_mask_rho_nan[outer_20_30m_mask_rho_nan_idx] = np.nan
# 30 - 60 m depth
outer_30_60m_mask_rho_nan_idx = np.where(outer_30_60m_mask_rho == 0.0)
outer_30_60m_mask_rho_nan = outer_30_60m_mask_rho.copy()
outer_30_60m_mask_rho_nan = outer_30_60m_mask_rho_nan.astype('float')
outer_30_60m_mask_rho_nan[outer_30_60m_mask_rho_nan_idx] = np.nan


# Now multiply by the mask to get the different regions 
# Make empty arrays to hold the values
# Kalikpik
kg_sed_0_10m_kal = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_kal = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_kal = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_kal = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Colville
kg_sed_0_10m_col = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_col = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_col = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_col = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Sagavanirktok
kg_sed_0_10m_sag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_sag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_sag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_sag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Fish Creek
kg_sed_0_10m_fis = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_fis = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_fis = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_fis = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# # Sakonwyak
# kg_sed_0_10m_sak = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_10_20m_sak = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_20_30m_sak = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_30_60m_sak = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Kuparuk
kg_sed_0_10m_kup = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_kup = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_kup = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_kup = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# # Putuligayuuk
# kg_sed_0_10m_put = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_10_20m_put = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_20_30m_put = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_30_60m_put = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Staines
kg_sed_0_10m_sta = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_sta = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_sta = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_sta = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Canning
kg_sed_0_10m_can = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_can = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_can = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_can = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Katakturuk
kg_sed_0_10m_kat = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_kat = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_kat = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_kat = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Hulahula
kg_sed_0_10m_hul = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_hul = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_hul = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_hul = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# Jago
kg_sed_0_10m_jag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_10_20m_jag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_20_30m_jag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
kg_sed_30_60m_jag = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# # Siksik
# kg_sed_0_10m_sik = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_10_20m_sik = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_20_30m_sik = np.empty((full_time_len,eta_rho_len,xi_rho_len))
# kg_sed_30_60m_sik = np.empty((full_time_len,eta_rho_len,xi_rho_len))


# Loop through time 
for tt in range(full_time_len):
    print('tt: ', tt, flush=True)
        
    # Kalikpik
    kg_sed_0_10m_kal[tt,:,:] = kg_sed_mud15_kal_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_kal[tt,:,:] = kg_sed_mud15_kal_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_kal[tt,:,:] = kg_sed_mud15_kal_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_kal[tt,:,:] = kg_sed_mud15_kal_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Colville
    kg_sed_0_10m_col[tt,:,:] = kg_sed_mud16_col_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_col[tt,:,:] = kg_sed_mud16_col_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_col[tt,:,:] = kg_sed_mud16_col_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_col[tt,:,:] = kg_sed_mud16_col_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Sagavanirktok
    kg_sed_0_10m_sag[tt,:,:] = kg_sed_mud17_sag_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_sag[tt,:,:] = kg_sed_mud17_sag_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_sag[tt,:,:] = kg_sed_mud17_sag_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_sag[tt,:,:] = kg_sed_mud17_sag_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Fish Creek
    kg_sed_0_10m_fis[tt,:,:] = kg_sed_mud18_fis_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_fis[tt,:,:] = kg_sed_mud18_fis_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_fis[tt,:,:] = kg_sed_mud18_fis_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_fis[tt,:,:] = kg_sed_mud18_fis_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # # Sakonowyak
    # kg_sed_0_10m_sak[tt,:,:] = kg_sed_mud19_sak_masked[tt,:,:]*inner_10m_mask_rho_nan
    # kg_sed_10_20m_sak[tt,:,:] = kg_sed_mud19_sak_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    # kg_sed_20_30m_sak[tt,:,:] = kg_sed_mud19_sak_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    # kg_sed_30_60m_sak[tt,:,:] = kg_sed_mud19_sak_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Kuparuk
    kg_sed_0_10m_kup[tt,:,:] = kg_sed_mud19_kup_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_kup[tt,:,:] = kg_sed_mud19_kup_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_kup[tt,:,:] = kg_sed_mud19_kup_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_kup[tt,:,:] = kg_sed_mud19_kup_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # # Putuligayuk
    # kg_sed_0_10m_put[tt,:,:] = kg_sed_mud21_put_masked[tt,:,:]*inner_10m_mask_rho_nan
    # kg_sed_10_20m_put[tt,:,:] = kg_sed_mud21_put_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    # kg_sed_20_30m_put[tt,:,:] = kg_sed_mud21_put_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    # kg_sed_30_60m_put[tt,:,:] = kg_sed_mud21_put_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Staines 
    kg_sed_0_10m_sta[tt,:,:] = kg_sed_mud20_sta_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_sta[tt,:,:] = kg_sed_mud20_sta_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_sta[tt,:,:] = kg_sed_mud20_sta_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_sta[tt,:,:] = kg_sed_mud20_sta_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Canning 
    kg_sed_0_10m_can[tt,:,:] = kg_sed_mud21_can_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_can[tt,:,:] = kg_sed_mud21_can_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_can[tt,:,:] = kg_sed_mud21_can_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_can[tt,:,:] = kg_sed_mud21_can_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Katakturuk
    kg_sed_0_10m_kat[tt,:,:] = kg_sed_mud22_kat_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_kat[tt,:,:] = kg_sed_mud22_kat_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_kat[tt,:,:] = kg_sed_mud22_kat_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_kat[tt,:,:] = kg_sed_mud22_kat_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Hulahua
    kg_sed_0_10m_hul[tt,:,:] = kg_sed_mud23_hul_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_hul[tt,:,:] = kg_sed_mud23_hul_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_hul[tt,:,:] = kg_sed_mud23_hul_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_hul[tt,:,:] = kg_sed_mud23_hul_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # Jago 
    kg_sed_0_10m_jag[tt,:,:] = kg_sed_mud24_jag_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_jag[tt,:,:] = kg_sed_mud24_jag_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_jag[tt,:,:] = kg_sed_mud24_jag_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_jag[tt,:,:] = kg_sed_mud24_jag_masked[tt,:,:]*outer_30_60m_mask_rho_nan
    # # Siksik
    # kg_sed_0_10m_sik[tt,:,:] = kg_sed_mud27_sik_masked[tt,:,:]*inner_10m_mask_rho_nan
    # kg_sed_10_20m_sik[tt,:,:] = kg_sed_mud27_sik_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    # kg_sed_20_30m_sik[tt,:,:] = kg_sed_mud27_sik_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    # kg_sed_30_60m_sik[tt,:,:] = kg_sed_mud27_sik_masked[tt,:,:]*outer_30_60m_mask_rho_nan


# Trim
# Kalikpik
kg_sed_0_10m_kal_masked_trimmed = kg_sed_0_10m_kal[:,:,c_west:-c_west]
kg_sed_10_20m_kal_masked_trimmed = kg_sed_10_20m_kal[:,:,c_west:-c_west]
kg_sed_20_30m_kal_masked_trimmed = kg_sed_20_30m_kal[:,:,c_west:-c_west]
kg_sed_30_60m_kal_masked_trimmed = kg_sed_30_60m_kal[:,:,c_west:-c_west]
# Colville
kg_sed_0_10m_col_masked_trimmed = kg_sed_0_10m_col[:,:,c_west:-c_west]
kg_sed_10_20m_col_masked_trimmed = kg_sed_10_20m_col[:,:,c_west:-c_west]
kg_sed_20_30m_col_masked_trimmed = kg_sed_20_30m_col[:,:,c_west:-c_west]
kg_sed_30_60m_col_masked_trimmed = kg_sed_30_60m_col[:,:,c_west:-c_west]
# Sagavanirktok
kg_sed_0_10m_sag_masked_trimmed = kg_sed_0_10m_sag[:,:,c_west:-c_west]
kg_sed_10_20m_sag_masked_trimmed = kg_sed_10_20m_sag[:,:,c_west:-c_west]
kg_sed_20_30m_sag_masked_trimmed = kg_sed_20_30m_sag[:,:,c_west:-c_west]
kg_sed_30_60m_sag_masked_trimmed = kg_sed_30_60m_sag[:,:,c_west:-c_west]
# Fish Creek
kg_sed_0_10m_fis_masked_trimmed = kg_sed_0_10m_fis[:,:,c_west:-c_west]
kg_sed_10_20m_fis_masked_trimmed = kg_sed_10_20m_fis[:,:,c_west:-c_west]
kg_sed_20_30m_fis_masked_trimmed = kg_sed_20_30m_fis[:,:,c_west:-c_west]
kg_sed_30_60m_fis_masked_trimmed = kg_sed_30_60m_fis[:,:,c_west:-c_west]
# # Sakonowyak
# kg_sed_0_10m_sak_masked_trimmed = kg_sed_0_10m_sak[:,:,c_west:-c_west]
# kg_sed_10_20m_sak_masked_trimmed = kg_sed_10_20m_sak[:,:,c_west:-c_west]
# kg_sed_20_30m_sak_masked_trimmed = kg_sed_20_30m_sak[:,:,c_west:-c_west]
# kg_sed_30_60m_sak_masked_trimmed = kg_sed_30_60m_sak[:,:,c_west:-c_west]
# Kuparuk
kg_sed_0_10m_kup_masked_trimmed = kg_sed_0_10m_kup[:,:,c_west:-c_west]
kg_sed_10_20m_kup_masked_trimmed = kg_sed_10_20m_kup[:,:,c_west:-c_west]
kg_sed_20_30m_kup_masked_trimmed = kg_sed_20_30m_kup[:,:,c_west:-c_west]
kg_sed_30_60m_kup_masked_trimmed = kg_sed_30_60m_kup[:,:,c_west:-c_west]
# # Putuligayuk
# kg_sed_0_10m_put_masked_trimmed = kg_sed_0_10m_put[:,:,c_west:-c_west]
# kg_sed_10_20m_put_masked_trimmed = kg_sed_10_20m_put[:,:,c_west:-c_west]
# kg_sed_20_30m_put_masked_trimmed = kg_sed_20_30m_put[:,:,c_west:-c_west]
# kg_sed_30_60m_put_masked_trimmed = kg_sed_30_60m_put[:,:,c_west:-c_west]
# Staines
kg_sed_0_10m_sta_masked_trimmed = kg_sed_0_10m_sta[:,:,c_west:-c_west]
kg_sed_10_20m_sta_masked_trimmed = kg_sed_10_20m_sta[:,:,c_west:-c_west]
kg_sed_20_30m_sta_masked_trimmed = kg_sed_20_30m_sta[:,:,c_west:-c_west]
kg_sed_30_60m_sta_masked_trimmed = kg_sed_30_60m_sta[:,:,c_west:-c_west]
# Canning
kg_sed_0_10m_can_masked_trimmed = kg_sed_0_10m_can[:,:,c_west:-c_west]
kg_sed_10_20m_can_masked_trimmed = kg_sed_10_20m_can[:,:,c_west:-c_west]
kg_sed_20_30m_can_masked_trimmed = kg_sed_20_30m_can[:,:,c_west:-c_west]
kg_sed_30_60m_can_masked_trimmed = kg_sed_30_60m_can[:,:,c_west:-c_west]
# Katakturuk
kg_sed_0_10m_kat_masked_trimmed = kg_sed_0_10m_kat[:,:,c_west:-c_west]
kg_sed_10_20m_kat_masked_trimmed = kg_sed_10_20m_kat[:,:,c_west:-c_west]
kg_sed_20_30m_kat_masked_trimmed = kg_sed_20_30m_kat[:,:,c_west:-c_west]
kg_sed_30_60m_kat_masked_trimmed = kg_sed_30_60m_kat[:,:,c_west:-c_west]
# Hulahula
kg_sed_0_10m_hul_masked_trimmed = kg_sed_0_10m_hul[:,:,c_west:-c_west]
kg_sed_10_20m_hul_masked_trimmed = kg_sed_10_20m_hul[:,:,c_west:-c_west]
kg_sed_20_30m_hul_masked_trimmed = kg_sed_20_30m_hul[:,:,c_west:-c_west]
kg_sed_30_60m_hul_masked_trimmed = kg_sed_30_60m_hul[:,:,c_west:-c_west]
# Jago
kg_sed_0_10m_jag_masked_trimmed = kg_sed_0_10m_jag[:,:,c_west:-c_west]
kg_sed_10_20m_jag_masked_trimmed = kg_sed_10_20m_jag[:,:,c_west:-c_west]
kg_sed_20_30m_jag_masked_trimmed = kg_sed_20_30m_jag[:,:,c_west:-c_west]
kg_sed_30_60m_jag_masked_trimmed = kg_sed_30_60m_jag[:,:,c_west:-c_west]
# # Siksik
# kg_sed_0_10m_sik_masked_trimmed = kg_sed_0_10m_sik[:,:,c_west:-c_west]
# kg_sed_10_20m_sik_masked_trimmed = kg_sed_10_20m_sik[:,:,c_west:-c_west]
# kg_sed_20_30m_sik_masked_trimmed = kg_sed_20_30m_sik[:,:,c_west:-c_west]
# kg_sed_30_60m_sik_masked_trimmed = kg_sed_30_60m_sik[:,:,c_west:-c_west]


# Before plotting and moving on, trim them all to just the time of the 
# model run that we trust (end on October 30)
# First find that time
print('full_time_len: ', full_time_len, flush=True)
print('Time step (720): ', time_steps[720], flush=True)
print('Time step (721): ', time_steps[721], flush=True)
print('Time step (722): ', time_steps[722], flush=True)
print('Time step (730): ', time_steps[730], flush=True)
print('Time step (736): ', time_steps[736], flush=True)
print('Time step (737): ', time_steps[737], flush=True)
print('Time step (738): ', time_steps[738], flush=True)
print('Time step (740): ', time_steps[740], flush=True)




# --------------------------------------------------------------------------------
# --------- Plot 1: Maps of kg Sediment by Region --------------------------------
# --------------------------------------------------------------------------------
# Now that things are split up by region, plot to make sure it worked
# Make a fake xy with the right resolution to be able to plot without the angle
x_rho_flat = np.arange(0,750*len(grid.x_rho[0,:]),750)
y_rho_flat = np.arange(0,600*len(grid.y_rho[:,0]),600)
# Prep the data by  ultiplying by the mask and trimming
# Trim 
x_rho_flat_trimmed = x_rho_flat[c_west:-c_west]
# Prep the data by  ultiplying by the mask and trimming
# Multiply by mask
h_masked = grid.h.values*grid.mask_rho.values*mask_rho_nan.nudge_mask_rho_nan
# Trim 
lon_rho_trimmed = grid.lon_rho[:,c_west:-c_west].values
lat_rho_trimmed = grid.lat_rho[:,c_west:-c_west].values
h_masked_trimmed = h_masked[:,c_west:-c_west]


# Make the figure
fig1, ax1 = plt.subplots(4, figsize=(22,21)) # (18,8) (26,12) (26,8) (26,10)

# Set the colormaps
cmap1 = cmocean.cm.turbid
cmap2 = cmocean.cm.delta # ero depo

# Set colorbar levels for bathymetry
lev1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Set colorbar levels for  ssc
lev2 = np.arange(0,0.05,0.001) # kg

# Kalikpik 0 - 10 m
ax1[0].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
               facecolor ='darkgray', alpha = 0.8)
ax1[0].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
               facecolor ='white', alpha = 0.8)
cs1 = ax1[0].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
                  kg_sed_0_10m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# Plot bathymetry contours
ax1[0].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# Label the plot
#ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
plt.setp(ax1[0].get_xticklabels(), visible=False)
#ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
ax1[0].set_ylabel('Y (km)', fontsize=fontsize)

# Kalikpik 10 - 20 m
ax1[1].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
               facecolor ='darkgray', alpha = 0.8)
ax1[1].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
               facecolor ='white', alpha = 0.8)
cs2 = ax1[1].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
                  kg_sed_10_20m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# Plot bathymetry contours
ax1[1].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# Label the plot
#ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
plt.setp(ax1[1].get_xticklabels(), visible=False)
#ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
ax1[1].set_ylabel('Y (km)', fontsize=fontsize)

# Kalikpik 20 - 30 m
ax1[2].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
               facecolor ='darkgray', alpha = 0.8)
ax1[2].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
               facecolor ='white', alpha = 0.8)
cs3 = ax1[2].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
                  kg_sed_20_30m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# Plot bathymetry contours
ax1[2].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# Label the plot
#ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
plt.setp(ax1[2].get_xticklabels(), visible=False)
#ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
ax1[2].set_ylabel('Y (km)', fontsize=fontsize)

# Kalikpik 30 - 60 m
ax1[3].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
               facecolor ='darkgray', alpha = 0.8)
ax1[3].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
               facecolor ='white', alpha = 0.8)
cs4 = ax1[3].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
                  kg_sed_30_60m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# Plot bathymetry contours
ax1[3].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# Label the plot
#ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
plt.setp(ax1[3].get_xticklabels(), visible=False)
#ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
ax1[3].set_ylabel('Y (km)', fontsize=fontsize)

cbar1 = plt.colorbar(cs4, orientation='vertical', ax=[ax1[0], ax1[1], ax1[2], ax1[3]]).set_label(label='Kalikpik Sediment (kg)', size=fontsize)



# Then assuming it worked, sum over each region to get a kg per time for each
# class in each region
# Kalikpik
kg_sed_0_10m_kal_masked_trimmed_sum = np.nansum(kg_sed_0_10m_kal_masked_trimmed, axis=(1,2))
kg_sed_10_20m_kal_masked_trimmed_sum = np.nansum(kg_sed_10_20m_kal_masked_trimmed, axis=(1,2))
kg_sed_20_30m_kal_masked_trimmed_sum = np.nansum(kg_sed_20_30m_kal_masked_trimmed, axis=(1,2))
kg_sed_30_60m_kal_masked_trimmed_sum = np.nansum(kg_sed_30_60m_kal_masked_trimmed, axis=(1,2))
# Colville
kg_sed_0_10m_col_masked_trimmed_sum = np.nansum(kg_sed_0_10m_col_masked_trimmed, axis=(1,2))
kg_sed_10_20m_col_masked_trimmed_sum = np.nansum(kg_sed_10_20m_col_masked_trimmed, axis=(1,2))
kg_sed_20_30m_col_masked_trimmed_sum = np.nansum(kg_sed_20_30m_col_masked_trimmed, axis=(1,2))
kg_sed_30_60m_col_masked_trimmed_sum = np.nansum(kg_sed_30_60m_col_masked_trimmed, axis=(1,2))
# Sagavanirktok
kg_sed_0_10m_sag_masked_trimmed_sum = np.nansum(kg_sed_0_10m_sag_masked_trimmed, axis=(1,2))
kg_sed_10_20m_sag_masked_trimmed_sum = np.nansum(kg_sed_10_20m_sag_masked_trimmed, axis=(1,2))
kg_sed_20_30m_sag_masked_trimmed_sum = np.nansum(kg_sed_20_30m_sag_masked_trimmed, axis=(1,2))
kg_sed_30_60m_sag_masked_trimmed_sum = np.nansum(kg_sed_30_60m_sag_masked_trimmed, axis=(1,2))
# Fish Creek
kg_sed_0_10m_fis_masked_trimmed_sum = np.nansum(kg_sed_0_10m_fis_masked_trimmed, axis=(1,2))
kg_sed_10_20m_fis_masked_trimmed_sum = np.nansum(kg_sed_10_20m_fis_masked_trimmed, axis=(1,2))
kg_sed_20_30m_fis_masked_trimmed_sum = np.nansum(kg_sed_20_30m_fis_masked_trimmed, axis=(1,2))
kg_sed_30_60m_fis_masked_trimmed_sum = np.nansum(kg_sed_30_60m_fis_masked_trimmed, axis=(1,2))
# # Sakonowyak
# kg_sed_0_10m_sak_masked_trimmed_sum = np.nansum(kg_sed_0_10m_sak_masked_trimmed, axis=(1,2))
# kg_sed_10_20m_sak_masked_trimmed_sum = np.nansum(kg_sed_10_20m_sak_masked_trimmed, axis=(1,2))
# kg_sed_20_30m_sak_masked_trimmed_sum = np.nansum(kg_sed_20_30m_sak_masked_trimmed, axis=(1,2))
# kg_sed_30_60m_sak_masked_trimmed_sum = np.nansum(kg_sed_30_60m_sak_masked_trimmed, axis=(1,2))
# Kuparuk
kg_sed_0_10m_kup_masked_trimmed_sum = np.nansum(kg_sed_0_10m_kup_masked_trimmed, axis=(1,2))
kg_sed_10_20m_kup_masked_trimmed_sum = np.nansum(kg_sed_10_20m_kup_masked_trimmed, axis=(1,2))
kg_sed_20_30m_kup_masked_trimmed_sum = np.nansum(kg_sed_20_30m_kup_masked_trimmed, axis=(1,2))
kg_sed_30_60m_kup_masked_trimmed_sum = np.nansum(kg_sed_30_60m_kup_masked_trimmed, axis=(1,2))
# # Putuligayuk
# kg_sed_0_10m_put_masked_trimmed_sum = np.nansum(kg_sed_0_10m_put_masked_trimmed, axis=(1,2))
# kg_sed_10_20m_put_masked_trimmed_sum = np.nansum(kg_sed_10_20m_put_masked_trimmed, axis=(1,2))
# kg_sed_20_30m_put_masked_trimmed_sum = np.nansum(kg_sed_20_30m_put_masked_trimmed, axis=(1,2))
# kg_sed_30_60m_put_masked_trimmed_sum = np.nansum(kg_sed_30_60m_put_masked_trimmed, axis=(1,2))
# Staines
kg_sed_0_10m_sta_masked_trimmed_sum = np.nansum(kg_sed_0_10m_sta_masked_trimmed, axis=(1,2))
kg_sed_10_20m_sta_masked_trimmed_sum = np.nansum(kg_sed_10_20m_sta_masked_trimmed, axis=(1,2))
kg_sed_20_30m_sta_masked_trimmed_sum = np.nansum(kg_sed_20_30m_sta_masked_trimmed, axis=(1,2))
kg_sed_30_60m_sta_masked_trimmed_sum = np.nansum(kg_sed_30_60m_sta_masked_trimmed, axis=(1,2))
# Canning
kg_sed_0_10m_can_masked_trimmed_sum = np.nansum(kg_sed_0_10m_can_masked_trimmed, axis=(1,2))
kg_sed_10_20m_can_masked_trimmed_sum = np.nansum(kg_sed_10_20m_can_masked_trimmed, axis=(1,2))
kg_sed_20_30m_can_masked_trimmed_sum = np.nansum(kg_sed_20_30m_can_masked_trimmed, axis=(1,2))
kg_sed_30_60m_can_masked_trimmed_sum = np.nansum(kg_sed_30_60m_can_masked_trimmed, axis=(1,2))
# Katakturuk
kg_sed_0_10m_kat_masked_trimmed_sum = np.nansum(kg_sed_0_10m_kat_masked_trimmed, axis=(1,2))
kg_sed_10_20m_kat_masked_trimmed_sum = np.nansum(kg_sed_10_20m_kat_masked_trimmed, axis=(1,2))
kg_sed_20_30m_kat_masked_trimmed_sum = np.nansum(kg_sed_20_30m_kat_masked_trimmed, axis=(1,2))
kg_sed_30_60m_kat_masked_trimmed_sum = np.nansum(kg_sed_30_60m_kat_masked_trimmed, axis=(1,2))
# Hulahula
kg_sed_0_10m_hul_masked_trimmed_sum = np.nansum(kg_sed_0_10m_hul_masked_trimmed, axis=(1,2))
kg_sed_10_20m_hul_masked_trimmed_sum = np.nansum(kg_sed_10_20m_hul_masked_trimmed, axis=(1,2))
kg_sed_20_30m_hul_masked_trimmed_sum = np.nansum(kg_sed_20_30m_hul_masked_trimmed, axis=(1,2))
kg_sed_30_60m_hul_masked_trimmed_sum = np.nansum(kg_sed_30_60m_hul_masked_trimmed, axis=(1,2))
# Jago
kg_sed_0_10m_jag_masked_trimmed_sum = np.nansum(kg_sed_0_10m_jag_masked_trimmed, axis=(1,2))
kg_sed_10_20m_jag_masked_trimmed_sum = np.nansum(kg_sed_10_20m_jag_masked_trimmed, axis=(1,2))
kg_sed_20_30m_jag_masked_trimmed_sum = np.nansum(kg_sed_20_30m_jag_masked_trimmed, axis=(1,2))
kg_sed_30_60m_jag_masked_trimmed_sum = np.nansum(kg_sed_30_60m_jag_masked_trimmed, axis=(1,2))
# # Siksik
# kg_sed_0_10m_sik_masked_trimmed_sum = np.nansum(kg_sed_0_10m_sik_masked_trimmed, axis=(1,2))
# kg_sed_10_20m_sik_masked_trimmed_sum = np.nansum(kg_sed_10_20m_sik_masked_trimmed, axis=(1,2))
# kg_sed_20_30m_sik_masked_trimmed_sum = np.nansum(kg_sed_20_30m_sik_masked_trimmed, axis=(1,2))
# kg_sed_30_60m_sik_masked_trimmed_sum = np.nansum(kg_sed_30_60m_sik_masked_trimmed, axis=(1,2))


# ---------------- Trim to Time we Trust --------------------
# Trim all of these variables to the time that we trust 
# Kalikpik
kg_sed_0_10m_kal_masked_trimmed_sum = kg_sed_0_10m_kal_masked_trimmed_sum[:738]
kg_sed_10_20m_kal_masked_trimmed_sum = kg_sed_10_20m_kal_masked_trimmed_sum[:738]
kg_sed_20_30m_kal_masked_trimmed_sum = kg_sed_20_30m_kal_masked_trimmed_sum[:738]
kg_sed_30_60m_kal_masked_trimmed_sum = kg_sed_30_60m_kal_masked_trimmed_sum[:738]
# Colville
kg_sed_0_10m_col_masked_trimmed_sum = kg_sed_0_10m_col_masked_trimmed_sum[:738]
kg_sed_10_20m_col_masked_trimmed_sum = kg_sed_10_20m_col_masked_trimmed_sum[:738]
kg_sed_20_30m_col_masked_trimmed_sum = kg_sed_20_30m_col_masked_trimmed_sum[:738]
kg_sed_30_60m_col_masked_trimmed_sum = kg_sed_30_60m_col_masked_trimmed_sum[:738]
# Sagavanirktok
kg_sed_0_10m_sag_masked_trimmed_sum = kg_sed_0_10m_sag_masked_trimmed_sum[:738]
kg_sed_10_20m_sag_masked_trimmed_sum = kg_sed_10_20m_sag_masked_trimmed_sum[:738]
kg_sed_20_30m_sag_masked_trimmed_sum = kg_sed_20_30m_sag_masked_trimmed_sum[:738]
kg_sed_30_60m_sag_masked_trimmed_sum = kg_sed_30_60m_sag_masked_trimmed_sum[:738]
# Fish Creek
kg_sed_0_10m_fis_masked_trimmed_sum = kg_sed_0_10m_fis_masked_trimmed_sum[:738]
kg_sed_10_20m_fis_masked_trimmed_sum = kg_sed_10_20m_fis_masked_trimmed_sum[:738]
kg_sed_20_30m_fis_masked_trimmed_sum = kg_sed_20_30m_fis_masked_trimmed_sum[:738]
kg_sed_30_60m_fis_masked_trimmed_sum = kg_sed_30_60m_fis_masked_trimmed_sum[:738]
# # Sakonowyak
# kg_sed_0_10m_sak_masked_trimmed_sum = kg_sed_0_10m_sak_masked_trimmed_sum[:738]
# kg_sed_10_20m_sak_masked_trimmed_sum = kg_sed_10_20m_sak_masked_trimmed_sum[:738]
# kg_sed_20_30m_sak_masked_trimmed_sum = kg_sed_20_30m_sak_masked_trimmed_sum[:738]
# kg_sed_30_60m_sak_masked_trimmed_sum = kg_sed_30_60m_sak_masked_trimmed_sum[:738]
# Kuparuk
kg_sed_0_10m_kup_masked_trimmed_sum = kg_sed_0_10m_kup_masked_trimmed_sum[:738]
kg_sed_10_20m_kup_masked_trimmed_sum = kg_sed_10_20m_kup_masked_trimmed_sum[:738]
kg_sed_20_30m_kup_masked_trimmed_sum = kg_sed_20_30m_kup_masked_trimmed_sum[:738]
kg_sed_30_60m_kup_masked_trimmed_sum = kg_sed_30_60m_kup_masked_trimmed_sum[:738]
# # Putuligayuk
# kg_sed_0_10m_put_masked_trimmed_sum = kg_sed_0_10m_put_masked_trimmed_sum[:738]
# kg_sed_10_20m_put_masked_trimmed_sum = kg_sed_10_20m_put_masked_trimmed_sum[:738]
# kg_sed_20_30m_put_masked_trimmed_sum = kg_sed_20_30m_put_masked_trimmed_sum[:738]
# kg_sed_30_60m_put_masked_trimmed_sum = kg_sed_30_60m_put_masked_trimmed_sum[:738]
# Staines
kg_sed_0_10m_sta_masked_trimmed_sum = kg_sed_0_10m_sta_masked_trimmed_sum[:738]
kg_sed_10_20m_sta_masked_trimmed_sum = kg_sed_10_20m_sta_masked_trimmed_sum[:738]
kg_sed_20_30m_sta_masked_trimmed_sum = kg_sed_20_30m_sta_masked_trimmed_sum[:738]
kg_sed_30_60m_sta_masked_trimmed_sum = kg_sed_30_60m_sta_masked_trimmed_sum[:738]
# Canning
kg_sed_0_10m_can_masked_trimmed_sum = kg_sed_0_10m_can_masked_trimmed_sum[:738]
kg_sed_10_20m_can_masked_trimmed_sum = kg_sed_10_20m_can_masked_trimmed_sum[:738]
kg_sed_20_30m_can_masked_trimmed_sum = kg_sed_20_30m_can_masked_trimmed_sum[:738]
kg_sed_30_60m_can_masked_trimmed_sum = kg_sed_30_60m_can_masked_trimmed_sum[:738]
# Katakturuk
kg_sed_0_10m_kat_masked_trimmed_sum = kg_sed_0_10m_kat_masked_trimmed_sum[:738]
kg_sed_10_20m_kat_masked_trimmed_sum = kg_sed_10_20m_kat_masked_trimmed_sum[:738]
kg_sed_20_30m_kat_masked_trimmed_sum = kg_sed_20_30m_kat_masked_trimmed_sum[:738]
kg_sed_30_60m_kat_masked_trimmed_sum = kg_sed_30_60m_kat_masked_trimmed_sum[:738]
# Hulahula
kg_sed_0_10m_hul_masked_trimmed_sum = kg_sed_0_10m_hul_masked_trimmed_sum[:738]
kg_sed_10_20m_hul_masked_trimmed_sum = kg_sed_10_20m_hul_masked_trimmed_sum[:738]
kg_sed_20_30m_hul_masked_trimmed_sum = kg_sed_20_30m_hul_masked_trimmed_sum[:738]
kg_sed_30_60m_hul_masked_trimmed_sum = kg_sed_30_60m_hul_masked_trimmed_sum[:738]
# Jago
kg_sed_0_10m_jag_masked_trimmed_sum = kg_sed_0_10m_jag_masked_trimmed_sum[:738]
kg_sed_10_20m_jag_masked_trimmed_sum = kg_sed_10_20m_jag_masked_trimmed_sum[:738]
kg_sed_20_30m_jag_masked_trimmed_sum = kg_sed_20_30m_jag_masked_trimmed_sum[:738]
kg_sed_30_60m_jag_masked_trimmed_sum = kg_sed_30_60m_jag_masked_trimmed_sum[:738]
# # Siksik
# kg_sed_0_10m_sik_masked_trimmed_sum = kg_sed_0_10m_sik_masked_trimmed_sum[:738]
# kg_sed_10_20m_sik_masked_trimmed_sum = kg_sed_10_20m_sik_masked_trimmed_sum[:738]
# kg_sed_20_30m_sik_masked_trimmed_sum = kg_sed_20_30m_sik_masked_trimmed_sum[:738]
# kg_sed_30_60m_sik_masked_trimmed_sum = kg_sed_30_60m_sik_masked_trimmed_sum[:738]

# Trim time steps to match
time_steps = time_steps[:738]



# --------------------------------------------------------------------------------
# --------- Plot 2: Time series of kg sediment by region per river ---------------
# --------------------------------------------------------------------------------
# Then work on plotting these on top of eachother (stacked histogram time series)
# Kalikpik
fig2, ax2 = plt.subplots(4, figsize=(28,14))
# Kalikpik
ax2[0].scatter(time_steps[0:50], kg_sed_0_10m_kal_masked_trimmed_sum[0:50], color='red', s=40)
#ax2[0].scatter(time_steps[0:50], kg_sed_10_20m_kal_masked_trimmed_sum[0:50], color='orange', s=40)
#ax2[0].scatter(time_steps[0:50], kg_sed_20_30m_kal_masked_trimmed_sum[0:50], color='gold', s=40)
#ax2[0].scatter(time_steps[0:50], kg_sed_30_60m_kal_masked_trimmed_sum[0:50], color='blue', s=40)
ax2[0].set_ylabel('kg', fontsize=fontsize)
ax2[0].set_title('Kalikpik', fontsize=fontsize)
plt.setp(ax2[0].get_xticklabels(), visible=False)
# Colville
ax2[1].scatter(time_steps[0:50], kg_sed_0_10m_col_masked_trimmed_sum[0:50], color='red', s=40)
#ax2[1].scatter(time_steps[0:50], kg_sed_10_20m_col_masked_trimmed_sum[0:50], color='orange', s=40)
#ax2[1].scatter(time_steps[0:50], kg_sed_20_30m_col_masked_trimmed_sum[0:50], color='gold', s=40)
#ax2[1].scatter(time_steps[0:50], kg_sed_30_60m_col_masked_trimmed_sum[0:50], color='blue', s=40)
ax2[1].set_ylabel('kg', fontsize=fontsize)
ax2[1].set_title('Colville', fontsize=fontsize)
plt.setp(ax2[1].get_xticklabels(), visible=False)
# Sagavanirktok
ax2[2].scatter(time_steps[0:50], kg_sed_0_10m_sag_masked_trimmed_sum[0:50], color='red', s=40)
#ax2[2].scatter(time_steps[0:50], kg_sed_10_20m_sag_masked_trimmed_sum[0:50], color='orange', s=40)
#ax2[2].scatter(time_steps[0:50], kg_sed_20_30m_sag_masked_trimmed_sum[0:50], color='gold', s=40)
#ax2[2].scatter(time_steps[0:50], kg_sed_30_60m_sag_masked_trimmed_sum[0:50], color='blue', s=40)
ax2[2].set_ylabel('kg', fontsize=fontsize)
ax2[2].set_title('Sagavanirktok', fontsize=fontsize)
plt.setp(ax2[2].get_xticklabels(), visible=False)
# Sagavanirktok
ax2[3].scatter(time_steps[0:50], kg_sed_0_10m_fis_masked_trimmed_sum[0:50], color='red', s=40)
#ax2[3].scatter(time_steps[0:50], kg_sed_10_20m_fis_masked_trimmed_sum[0:50], color='orange', s=40)
#ax2[3].scatter(time_steps[0:50], kg_sed_20_30m_fis_masked_trimmed_sum[0:50], color='gold', s=40)
#ax2[3].scatter(time_steps[0:50], kg_sed_30_60m_fis_masked_trimmed_sum[0:50], color='blue', s=40)
ax2[3].set_ylabel('kg', fontsize=fontsize)
ax2[3].set_title('Fish Creek', fontsize=fontsize)
ax2[3].set_xlabel('Time', fontsize=fontsize)



# --------------------------------------------------------------------------------
# --------- Plot 3: Stacked time series of river sediment for Kalikpik -----------
# --------------------------------------------------------------------------------
# Try making a plot where they are stacked on top 
# First need to make it into a dataframe?
# Kalikpik
df_kal = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_kal_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_kal_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_kal_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_kal_masked_trimmed_sum},
    index=pd.to_datetime(time_steps))

xticks = ['2020-07-01', '2020-07-15', '2020-08-01', '2020-08-15']

# Make the figure - version using pandas
fig3, ax3 = plt.subplots(figsize=(18,5))
# Set the colors
colors3 = ['royalblue', 'red', 'gold', 'purple']
# Format the x-axis 
ax3.xaxis.set_major_formatter('%Y-%m-%d')
df_kal.plot.area(ax=ax3, fontsize=fontsize-5, color=colors3, xticks=xticks)
ax3.legend(fontsize=fontsize-10)
ax3.axhline(0.4)
#ax3.xaxis.set_major_formatter("%Y-%m-%d")
ax3.set_xlabel('Time', fontsize=fontsize-8)
ax3.set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax3.set_title('Kalikpik River Sediment Mass on Shelf', fontsize=fontsize-5)
#plt.setp(ax3.get_xticklabels(), visible=False)
#plt.xticks([])


# Make the figure - version using matplotlib
fig3, ax3 = plt.subplots(figsize=(18,5))
# Set the colors
colors3 = ['royalblue', 'red', 'gold', 'purple']
# Format the x-axis 
#ax3.xaxis.set_major_formatter('%Y-%m-%d')
ax3.stackplot(df_kal.index, df_kal['0 - 10 m'], df_kal['10 - 20 m'], df_kal['20 - 30 m'], 
              df_kal['30 - 60 m'], labels=df_kal.columns, colors=colors3)
ax3.legend(fontsize=fontsize-10)
#ax3.axhline(0.4)
#ax3.xaxis.set_major_formatter("%Y-%m-%d")
ax3.set_xlabel('Time', fontsize=fontsize-8)
ax3.set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax3.set_title('Kalikpik River Sediment Mass on Shelf', fontsize=fontsize-5)
#plt.setp(ax3.get_xticklabels(), visible=False)
plt.xticks(['2020-07-01', '2020-07-15', '2020-08-01', '2020-08-15'])


# --------------------------------------------------------------------------------
# ----- Plot 4: Stacked time series of river sediment for All Rivers -------------
# --------------------------------------------------------------------------------
# Make a subplot version of this for all of the different rivers 
# (remember, not all of the output data is here yet)
 
# Make dataframes for each river 
# Colville
df_col = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_col_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_col_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_col_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_col_masked_trimmed_sum},
    index=time_steps)
# Sagavanirktok
df_sag = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_sag_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_sag_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_sag_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_sag_masked_trimmed_sum},
    index=time_steps)
# Fish Creek
df_fis = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_fis_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_fis_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_fis_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_fis_masked_trimmed_sum},
    index=time_steps)
# # Sakonowyak
# df_sak = pd.DataFrame({
#     '0 - 10 m': kg_sed_0_10m_sak_masked_trimmed_sum,
#     '10 - 20 m': kg_sed_10_20m_sak_masked_trimmed_sum,
#     '20 - 30 m': kg_sed_20_30m_sak_masked_trimmed_sum,
#     '30 - 60 m': kg_sed_30_60m_sak_masked_trimmed_sum},
#     index=time_steps)
# Kuparik
df_kup = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_kup_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_kup_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_kup_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_kup_masked_trimmed_sum},
    index=time_steps)
# # Putuligayuk
# df_put = pd.DataFrame({
#     '0 - 10 m': kg_sed_0_10m_put_masked_trimmed_sum,
#     '10 - 20 m': kg_sed_10_20m_put_masked_trimmed_sum,
#     '20 - 30 m': kg_sed_20_30m_put_masked_trimmed_sum,
#     '30 - 60 m': kg_sed_30_60m_put_masked_trimmed_sum},
#     index=time_steps)
# Staines
df_sta = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_sta_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_sta_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_sta_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_sta_masked_trimmed_sum},
    index=time_steps)
# Canning
df_can = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_can_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_can_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_can_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_can_masked_trimmed_sum},
    index=time_steps)
# Katakturuk
df_kat = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_kat_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_kat_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_kat_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_kat_masked_trimmed_sum},
    index=time_steps)
# Hulahula
df_hul = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_hul_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_hul_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_hul_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_hul_masked_trimmed_sum},
    index=time_steps)
# Jago
df_jag = pd.DataFrame({
    '0 - 10 m': kg_sed_0_10m_jag_masked_trimmed_sum,
    '10 - 20 m': kg_sed_10_20m_jag_masked_trimmed_sum,
    '20 - 30 m': kg_sed_20_30m_jag_masked_trimmed_sum,
    '30 - 60 m': kg_sed_30_60m_jag_masked_trimmed_sum},
    index=time_steps)
# # Siksik
# df_sik = pd.DataFrame({
#     '0 - 10 m': kg_sed_0_10m_sik_masked_trimmed_sum,
#     '10 - 20 m': kg_sed_10_20m_sik_masked_trimmed_sum,
#     '20 - 30 m': kg_sed_20_30m_sik_masked_trimmed_sum,
#     '30 - 60 m': kg_sed_30_60m_sik_masked_trimmed_sum},
#     index=time_steps)


# Now make a subplot and plot them
fig4, ax4 = plt.subplots(3,4, figsize=(36,20)) # (28,20) (width, height)

# Set the colors for the different regions 
colors4 = ['royalblue', 'red', 'gold', 'purple']

# Plot the rivers
# Kalikpik 
df_kal.plot.area(ax=ax4[0,0], color=colors4, legend=False, sharex=True)
#ax4[0,0].legend(fontsize=fontsize-10)
#ax4[0,0].set_xlabel('Time', fontsize=fontsize-8)
#ax4[0,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[0,0].set_title('Kalikpik River', fontsize=fontsize-3)
plt.setp(ax4[0,0].get_xticklabels(), visible=False)

# Colville
df_col.plot.area(ax=ax4[0,1], color=colors4, legend=False, sharex=True)
#ax4[0,1].legend(fontsize=fontsize-10)
#ax4[0,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[0,1].set_title('Colville River', fontsize=fontsize-3)
plt.setp(ax4[0,1].get_xticklabels(), visible=False)

# Sagavanirktok
df_sag.plot.area(ax=ax4[0,2], color=colors4, legend=False, sharex=True)
#ax4[0,2].legend(fontsize=fontsize-10)
#ax4[0,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[0,2].set_title('Sagavanirktok River', fontsize=fontsize-3)
plt.setp(ax4[0,2].get_xticklabels(), visible=False)

# Fish Creek
df_fis.plot.area(ax=ax4[0,3], color=colors4, legend=False, sharex=True)
#ax4[1,0].legend(fontsize=fontsize-10)
#ax4[1,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[0,3].set_title('Fish Creek', fontsize=fontsize-3)
plt.setp(ax4[0,3].get_xticklabels(), visible=False)

# # Sakonowyak
# df_sak.plot.area(ax=ax4[1,0], color=colors4, legend=False, sharex=True)
# #ax4[1,1].legend(fontsize=fontsize-10)
# #ax4[1,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax4[1,0].set_title('Sakonowyak River', fontsize=fontsize-3)
# plt.setp(ax4[1,0].get_xticklabels(), visible=False)

# Kuparuk
df_kup.plot.area(ax=ax4[1,0], color=colors4, legend=False, sharex=True)
#ax4[1,2].legend(fontsize=fontsize-10)
#ax4[1,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[1,0].set_title('Kuparik River', fontsize=fontsize-3)
plt.setp(ax4[1,0].get_xticklabels(), visible=False)

# # Putuligayuk
# df_put.plot.area(ax=ax4[1,2], color=colors4, legend=False, sharex=True)
# #ax4[2,0].legend(fontsize=fontsize-10)
# #ax4[2,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax4[1,2].set_title('Putuligayuk River', fontsize=fontsize-3)
# plt.setp(ax4[1,2].get_xticklabels(), visible=False)

# Staines
df_sta.plot.area(ax=ax4[1,1], color=colors4, legend=False, sharex=True)
#ax4[2,1].legend(fontsize=fontsize-10)
#ax4[2,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[1,1].set_title('Staines River', fontsize=fontsize-3)
plt.setp(ax4[1,1].get_xticklabels(), visible=False)

# Canning
df_can.plot.area(ax=ax4[1,2], color=colors4, legend=False, sharex=True)
#ax4[2,2].legend(fontsize=fontsize-10)
#ax4[2,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[1,2].set_title('Canning River', fontsize=fontsize-3)
plt.setp(ax4[1,2].get_xticklabels(), visible=False)

# Katakturuk
df_kat.plot.area(ax=ax4[1,3], color=colors4, legend=False)
#ax4[3,0].legend(fontsize=fontsize-10)
#ax4[3,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[1,3].set_title('Katakturuk River', fontsize=fontsize-3)
plt.setp(ax4[2,1].get_xticklabels(), visible=True, fontsize=fontsize-3)
ax4[1,3].set_xlabel('Time', fontsize=fontsize+5)

# Hulahula - doing something weird with negative SSC...
df_hul.plot.area(ax=ax4[2,0], color=colors4, legend=False)
#ax4[2,2].legend(fontsize=fontsize-10)
#ax4[2,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[2,0].set_title('Hulahula River', fontsize=fontsize-5)
plt.setp(ax4[2,0].get_xticklabels(), visible=True, fontsize=fontsize-5)
ax4[2,0].set_xlabel('Time', fontsize=fontsize+5)

# Jago
df_jag.plot.area(ax=ax4[2,1], color=colors4, legend=False)
#ax4[3,2].legend(fontsize=fontsize-10)
#ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax4[2,1].set_title('Jago River', fontsize=fontsize-3)
plt.setp(ax4[2,1].get_xticklabels(), visible=True, fontsize=fontsize-3)
ax4[2,1].set_xlabel('Time', fontsize=fontsize+5)

# # Siksik
# df_sik.plot.area(ax=ax4[3,0], color=colors4, legend=False)
# #ax4[3,2].legend(fontsize=fontsize-10)
# #ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax4[3,0].set_title('Siksik River', fontsize=fontsize-3)
# plt.setp(ax4[3,0].get_xticklabels(), visible=True, fontsize=fontsize-3)
# ax4[3,0].set_xlabel('Time', fontsize=fontsize+5)

# Set universal axes labels 
#fig4.supxlabel('Time', fontsize=fontsize, labelpad=2)
#fig4.supylabel('Sediment Mass (kg)', fontsize=fontsize, labelpad=0.2)

# Hide the other axes
ax4[2,2].axis('off')
ax4[2,3].axis('off')
#ax4[3,3].axis('off')

# Label the plot
#fig1.text(0.5, 0.03, "X (km)", ha="center", va="center", fontsize=fontsize+5)
#fig1.text(0.07, 0.55, "Y (km)", ha="center", va="center", rotation=0, fontsize=fontsize+5)
#fig1.text(0.5, 0.92, 'Net Deposition (kg)', ha="center", va="center", rotation=0, fontsize=fontsize+5)

#fig4.text(0.5, 0.03, "Time", ha="center", va="center", fontsize=fontsize+5)
fig4.text(0.07, 0.53, "Sediment \nMass \n(kg)", ha="center", va="center", rotation=0, fontsize=fontsize+5)
fig4.text(0.5, 0.92, 'River Suspended Sediment Mass by Depth & Time (kg)', ha="center", va="center", rotation=0, fontsize=fontsize+5)

# Set universal legend 
# Put a legend for the rivers
#ax4[3,0].legend(fontsize=fontsize-4, loc='center left', ncol=1, 
 #             labelspacing=0.1,  bbox_to_anchor=(1.05, 0.3))
ax4[2,3].legend(fontsize=fontsize+2, loc='center left', ncol=1, 
              labelspacing=0.1,  bbox_to_anchor=(0.25, -0.75))

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.22, wspace=0.35) #0.08 # (0.22, 0.25)

# Save the figure
# Aggregated
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_allrivs_first_2020_aggregated_dbsed0001_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)
# Unaggregated
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_allrivs_first_2020_unaggregated_dbsed0001_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)



# # --------------------------------------------------------------------------------
# # ----- Plot 5: Stacked time series of river sediment for All Rivers -------------
# # -------------------- with river discharge plotted on top -----------------------
# # --------------------------------------------------------------------------------
# # Make the same plot as before but with the river discharge plotted on top, too
# # with the other y-axis 
# # soooooo earlier this was working just fine with the river discharge showing up
# # on top of the plot but now it is randomly not working which is very annoying 
# # but maybe we will figure it out later...
# # (all stopped working when I added the other model output files in the middle into 
# # the script/folder of model output)

# # Now make a subplot and plot them
# fig5, ax5 = plt.subplots(3,4, figsize=(34,20)) # 28,20

# # Set the colors for the different regions 
# colors5 = ['royalblue', 'red', 'gold', 'purple']

# # Plot the rivers
# # Kalikpik 
# #df_kal.plot.area(ax=ax5[0,0], color=colors5, legend=False, sharex=True)
# ax5[0,0].stackplot(df_kal.index, df_kal['0 - 10 m'], df_kal['10 - 20 m'], df_kal['20 - 30 m'], 
#               df_kal['30 - 60 m'], labels=df_kal.columns, colors=colors5)
# plt.setp(ax5[0,0].get_yticklabels(), color='royalblue')
# ax5_00 = ax5[0,0].twinx()
# ax5_00.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,0].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_00.get_yticklabels(), color='green')
# #ax4[0,0].legend(fontsize=fontsize-10)
# #ax4[0,0].set_xlabel('Time', fontsize=fontsize-8)
# #ax4[0,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[0,0].set_title('Kalikpik River', fontsize=fontsize-3)
# plt.setp(ax5[0,0].get_xticklabels(), visible=False)

# # Colville
# #df_col.plot.area(ax=ax5[0,1], color=colors5, legend=False, sharex=True)
# ax5[0,1].stackplot(df_col.index, df_col['0 - 10 m'], df_col['10 - 20 m'], df_col['20 - 30 m'], 
#               df_col['30 - 60 m'], labels=df_col.columns, colors=colors5)
# plt.setp(ax5[0,1].get_yticklabels(), color='royalblue')
# ax5_01 = ax5[0,1].twinx()
# # Get Colville discharge 
# dis_col = np.sum(river_frc.river_transport[:,1:7], axis=1)
# ax5_01.plot(river_frc.river_time[:40].values, dis_col[:40], 
#             color='green', linewidth=4)
# plt.setp(ax5_01.get_yticklabels(), color='green')
# #ax4[0,1].legend(fontsize=fontsize-10)
# #ax4[0,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[0,1].set_title('Colville River', fontsize=fontsize-3)
# plt.setp(ax5[0,1].get_xticklabels(), visible=False)

# # Sagavanirktok
# #df_sag.plot.area(ax=ax5[0,2], color=colors5, legend=False, sharex=True)
# ax5[0,2].stackplot(df_sag.index, df_sag['0 - 10 m'], df_sag['10 - 20 m'], df_sag['20 - 30 m'], 
#               df_sag['30 - 60 m'], labels=df_sag.columns, colors=colors5)
# plt.setp(ax5[0,2].get_yticklabels(), color='royalblue')
# ax5_02 = ax5[0,2].twinx()
# # Get discharge 
# dis_sag = np.sum(river_frc.river_transport[:,7:10], axis=1)
# ax5_02.plot(river_frc.river_time[:40].values, dis_sag[:40], 
#             color='green', linewidth=4)
# plt.setp(ax5_02.get_yticklabels(), color='green')
# #ax4[0,2].legend(fontsize=fontsize-10)
# #ax4[0,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[0,2].set_title('Sagavanirktok River', fontsize=fontsize-3)
# plt.setp(ax5[0,2].get_xticklabels(), visible=False)

# # Fish Creek
# #df_fis.plot.area(ax=ax5[1,0], color=colors5, legend=False, sharex=True)
# ax5[0,3].stackplot(df_fis.index, df_fis['0 - 10 m'], df_fis['10 - 20 m'], df_fis['20 - 30 m'], 
#               df_fis['30 - 60 m'], labels=df_fis.columns, colors=colors5)
# plt.setp(ax5[0,3].get_yticklabels(), color='royalblue')
# ax5_10 = ax5[0,3].twinx()
# ax5_10.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,10].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_10.get_yticklabels(), color='green')
# #ax4[1,0].legend(fontsize=fontsize-10)
# #ax4[1,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[0,3].set_title('Fish Creek', fontsize=fontsize-3)
# plt.setp(ax5[0,3].get_xticklabels(), visible=False)

# # # Sakonowyak
# # #df_sak.plot.area(ax=ax5[1,1], color=colors5, legend=False, sharex=True)
# # ax5[1,1].stackplot(df_sak.index, df_sak['0 - 10 m'], df_sak['10 - 20 m'], df_sak['20 - 30 m'], 
# #               df_sak['30 - 60 m'], labels=df_sak.columns, colors=colors5)
# # plt.setp(ax5[1,1].get_yticklabels(), color='royalblue')
# # ax5_11 = ax5[1,1].twinx()
# # ax5_11.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,11].values, 
# #             color='green', linewidth=4)
# # plt.setp(ax5_11.get_yticklabels(), color='green')
# # #ax4[1,1].legend(fontsize=fontsize-10)
# # #ax4[1,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# # ax5[1,1].set_title('Sakonowyak River', fontsize=fontsize-3)
# # plt.setp(ax5[1,1].get_xticklabels(), visible=False)

# # Kuparuk
# #df_kup.plot.area(ax=ax5[1,2], color=colors5, legend=False, sharex=True)
# ax5[1,0].stackplot(df_kup.index, df_kup['0 - 10 m'], df_kup['10 - 20 m'], df_kup['20 - 30 m'], 
#               df_kup['30 - 60 m'], labels=df_kup.columns, colors=colors5)
# plt.setp(ax5[1,0].get_yticklabels(), color='royalblue')
# ax5_12 = ax5[1,0].twinx()
# # Get discharge 
# dis_kup = np.sum(river_frc.river_transport[:,12], axis=1)
# ax5_12.plot(river_frc.river_time[:40].values, dis_kup[:40], 
#             color='green', linewidth=4)
# plt.setp(ax5_12.get_yticklabels(), color='green')
# #ax4[1,2].legend(fontsize=fontsize-10)
# #ax4[1,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[1,0].set_title('Kuparik River', fontsize=fontsize-3)
# plt.setp(ax5[1,0].get_xticklabels(), visible=False)

# # # Putuligayuk
# # #df_put.plot.area(ax=ax5[2,0], color=colors5, legend=False, sharex=True)
# # ax5[2,0].stackplot(df_put.index, df_put['0 - 10 m'], df_put['10 - 20 m'], df_put['20 - 30 m'], 
# #               df_put['30 - 60 m'], labels=df_put.columns, colors=colors5)
# # plt.setp(ax5[2,0].get_yticklabels(), color='royalblue')
# # ax5_20 = ax5[2,0].twinx()
# # ax5_20.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,14].values, 
# #             color='green', linewidth=4)
# # plt.setp(ax5_20.get_yticklabels(), color='green')
# # #ax4[2,0].legend(fontsize=fontsize-10)
# # #ax4[2,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# # ax5[2,0].set_title('Putuligayuk River', fontsize=fontsize-3)
# # plt.setp(ax5[2,0].get_xticklabels(), visible=False)

# # Staines
# #df_sta.plot.area(ax=ax5[2,1], color=colors5, legend=False, sharex=True)
# ax5[1,1].stackplot(df_sta.index, df_sta['0 - 10 m'], df_sta['10 - 20 m'], df_sta['20 - 30 m'], 
#               df_sta['30 - 60 m'], labels=df_sta.columns, colors=colors5)
# plt.setp(ax5[1,1].get_yticklabels(), color='royalblue')
# ax5_21 = ax5[1,1].twinx()
# ax5_21.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,13].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_21.get_yticklabels(), color='green')
# #ax4[2,1].legend(fontsize=fontsize-10)
# #ax4[2,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[1,1].set_title('Staines River', fontsize=fontsize-3)
# plt.setp(ax5[1,1].get_xticklabels(), visible=False)

# # Canning
# #df_can.plot.area(ax=ax5[2,2], color=colors5, legend=False, sharex=True)
# ax5[1,2].stackplot(df_can.index, df_can['0 - 10 m'], df_can['10 - 20 m'], df_can['20 - 30 m'], 
#               df_can['30 - 60 m'], labels=df_can.columns, colors=colors5)
# plt.setp(ax5[1,2].get_yticklabels(), color='royalblue')
# ax5_22 = ax5[1,2].twinx()
# ax5_22.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,14].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_22.get_yticklabels(), color='green')
# #ax4[2,2].legend(fontsize=fontsize-10)
# #ax4[2,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[1,2].set_title('Canning River', fontsize=fontsize-3)
# plt.setp(ax5[1,2].get_xticklabels(), visible=False)

# # Katakturuk
# #df_kat.plot.area(ax=ax5[3,0], color=colors5, legend=False)
# ax5[1,3].stackplot(df_kat.index, df_kat['0 - 10 m'], df_kat['10 - 20 m'], df_kat['20 - 30 m'], 
#               df_kat['30 - 60 m'], labels=df_kat.columns, colors=colors5)
# plt.setp(ax5[1,3].get_yticklabels(), color='royalblue')
# ax5_30 = ax5[1,3].twinx()
# ax5_30.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,15].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_30.get_yticklabels(), color='green')
# #ax4[3,0].legend(fontsize=fontsize-10)
# #ax4[3,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[1,3].set_title('Katakturuk River', fontsize=fontsize-3)
# plt.setp(ax5[1,3].get_xticklabels(), visible=True, fontsize=fontsize-3)
# plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# # Hulahula - doing something weird with negative SSC...
# #df_hul.plot.area(ax=ax4[3,1], color=colors4)
# #ax4[3,1].legend(fontsize=fontsize-10)
# #ax4[3,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# #ax4[3,1].set_title('Hulahula River', fontsize=fontsize-5)
# #plt.setp(ax4[3,1].get_xticklabels(), visible=True, fontsize=fontsize-5)

# # Jago
# #df_jag.plot.area(ax=ax5[3,1], color=colors5, legend=False)
# ax5[2,0].stackplot(df_jag.index, df_jag['0 - 10 m'], df_jag['10 - 20 m'], df_jag['20 - 30 m'], 
#               df_jag['30 - 60 m'], labels=df_jag.columns, colors=colors5)
# plt.setp(ax5[2,0].get_yticklabels(), color='royalblue')
# ax5_31 = ax5[2,0].twinx()
# ax5_31.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,16].values, 
#             color='green', linewidth=4)
# plt.setp(ax5_31.get_yticklabels(), color='green')
# #ax4[3,1].legend(fontsize=fontsize-10)
# #ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax5[2,0].set_title('Jago River', fontsize=fontsize-3)
# plt.setp(ax5[2,0].get_xticklabels(), visible=True, fontsize=fontsize-3)
# plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# # # Siksik
# # #df_sik.plot.area(ax=ax5[3,2], color=colors5, legend=False) #, xticks=['2019-07-01', '2019-07-15', '2019-08-01', '2019-08-15']
# # ax5[3,2].stackplot(df_sik.index, df_sik['0 - 10 m'], df_sik['10 - 20 m'], df_sik['20 - 30 m'], 
# #               df_sik['30 - 60 m'], labels=df_sik.columns, colors=colors5)
# # plt.setp(ax5[3,2].get_yticklabels(), color='royalblue')
# # ax5_32 = ax5[3,2].twinx()
# # ax5_32.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,20].values, 
# #             color='green', linewidth=4)
# # plt.setp(ax5_32.get_yticklabels(), color='green')
# # #ax4[3,2].legend(fontsize=fontsize-10)
# # #ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# # ax5[3,2].set_title('Siksik River', fontsize=fontsize-3)
# # plt.setp(ax5[3,2].get_xticklabels(), visible=True, fontsize=fontsize-3)
# # plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# # Set universal axes labels 
# #fig4.supxlabel('Time', fontsize=fontsize, labelpad=2)
# #fig4.supylabel('Sediment Mass (kg)', fontsize=fontsize, labelpad=0.2)

# fig5.text(0.5, 0.03, "Time", ha="center", va="center", fontsize=fontsize+5)
# fig5.text(0.05, 0.55, "Sediment \nMass (kg)", ha="center", va="center", rotation=0, fontsize=fontsize+5, color='royalblue')
# fig5.text(0.97, 0.55, "River \nDischarge \n(m\u00b3/s)", ha="center", va="center", rotation=0, fontsize=fontsize+5, color='green')
# fig5.text(0.5, 0.92, 'River Sediment Mass by Depth & Time (kg)', ha="center", va="center", rotation=0, fontsize=fontsize+5)

# # Set universal legend 
# # Put a legend for the rivers
# ax5[3,2].legend(fontsize=fontsize-4, loc='lower left', ncol=1, 
#               labelspacing=0.1,  bbox_to_anchor=(1.17, 0.1))

# # Adjust spacing between subplots
# plt.subplots_adjust(hspace=0.22, wspace=0.45) #0.08

# # Save the plot
# # Aggregated
# plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_nohula_discharge_2020_aggregated_dbsed0001_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_nohula_discharge_2020_unaggregated_dbsed0001_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# --------------------------------------------------------------------------------
# ----- Plot 6: Stacked time series of river sediment for All Rivers -------------
# -------------------- using other method -----------------------
# --------------------------------------------------------------------------------
# Same as the first plot with all the rivers but plotted using the same method 
# as the second plot to avoid the weird stacking issue


# Now make a subplot and plot them
fig6, ax6 = plt.subplots(3,4, figsize=(36,20))

# Set the colors for the different regions 
colors6 = ['royalblue', 'red', 'gold', 'purple']

# Plot the rivers
# Kalikpik 
#df_kal.plot.area(ax=ax5[0,0], color=colors5, legend=False, sharex=True)
ax6[0,0].stackplot(df_kal.index, df_kal['0 - 10 m'], df_kal['10 - 20 m'], df_kal['20 - 30 m'], 
              df_kal['30 - 60 m'], labels=df_kal.columns, colors=colors6)
#plt.setp(ax6[0,0].get_yticklabels(), color='royalblue')
#ax6_00 = ax6[0,0].twinx()
#ax6_00.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,0].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_00.get_yticklabels(), color='green')
#ax4[0,0].legend(fontsize=fontsize-10)
#ax4[0,0].set_xlabel('Time', fontsize=fontsize-8)
#ax4[0,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[0,0].set_title('Kalikpik River', fontsize=fontsize-3)
plt.setp(ax6[0,0].get_xticklabels(), visible=False)
#ax6[0,0].set_ylim([0,3e6])

# Colville
#df_col.plot.area(ax=ax5[0,1], color=colors5, legend=False, sharex=True)
ax6[0,1].stackplot(df_col.index, df_col['0 - 10 m'], df_col['10 - 20 m'], df_col['20 - 30 m'], 
              df_col['30 - 60 m'], labels=df_col.columns, colors=colors6)
#plt.setp(ax6[0,1].get_yticklabels(), color='royalblue')
#ax6_01 = ax6[0,1].twinx()
# Get Colville discharge 
#dis_col = np.sum(river_frc.river_transport[:,1:7], axis=1)
#ax6_01.plot(river_frc.river_time[:40].values, dis_col[:40], 
 #           color='green', linewidth=4)
#plt.setp(ax6_01.get_yticklabels(), color='green')
#ax4[0,1].legend(fontsize=fontsize-10)
#ax4[0,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[0,1].set_title('Colville River', fontsize=fontsize-3)
plt.setp(ax6[0,1].get_xticklabels(), visible=False)
#ax6[0,1].set_ylim([0,3e6])

# Sagavanirktok
#df_sag.plot.area(ax=ax5[0,2], color=colors5, legend=False, sharex=True)
ax6[0,2].stackplot(df_sag.index, df_sag['0 - 10 m'], df_sag['10 - 20 m'], df_sag['20 - 30 m'], 
              df_sag['30 - 60 m'], labels=df_sag.columns, colors=colors6)
#plt.setp(ax6[0,2].get_yticklabels(), color='royalblue')
#ax6_02 = ax6[0,2].twinx()
# Get discharge 
#dis_sag = np.sum(river_frc.river_transport[:,7:10], axis=1)
#ax6_02.plot(river_frc.river_time[:40].values, dis_sag[:40], 
 #           color='green', linewidth=4)
#plt.setp(ax6_02.get_yticklabels(), color='green')
#ax4[0,2].legend(fontsize=fontsize-10)
#ax4[0,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[0,2].set_title('Sagavanirktok River', fontsize=fontsize-3)
plt.setp(ax6[0,2].get_xticklabels(), visible=False)
#ax6[0,2].set_ylim([0,3e6])

# Fish Creek
#df_fis.plot.area(ax=ax5[1,0], color=colors5, legend=False, sharex=True)
ax6[0,3].stackplot(df_fis.index, df_fis['0 - 10 m'], df_fis['10 - 20 m'], df_fis['20 - 30 m'], 
              df_fis['30 - 60 m'], labels=df_fis.columns, colors=colors6)
#plt.setp(ax6[1,0].get_yticklabels(), color='royalblue')
#ax6_10 = ax6[1,0].twinx()
#ax6_10.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,10].values, 
 #           color='green', linewidth=4)
#plt.setp(ax6_10.get_yticklabels(), color='green')
#ax4[1,0].legend(fontsize=fontsize-10)
#ax4[1,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[0,3].set_title('Fish Creek', fontsize=fontsize-3)
plt.setp(ax6[0,3].get_xticklabels(), visible=False)
#ax6[0,3].set_ylim([0,3e6])

# # Sakonowyak
# #df_sak.plot.area(ax=ax5[1,1], color=colors5, legend=False, sharex=True)
# ax6[1,0].stackplot(df_sak.index, df_sak['0 - 10 m'], df_sak['10 - 20 m'], df_sak['20 - 30 m'], 
#               df_sak['30 - 60 m'], labels=df_sak.columns, colors=colors6)
# #plt.setp(ax5[1,1].get_yticklabels(), color='royalblue')
# #ax5_11 = ax5[1,1].twinx()
# #ax5_11.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,11].values, 
#  #           color='green', linewidth=4)
# #plt.setp(ax5_11.get_yticklabels(), color='green')
# #ax4[1,1].legend(fontsize=fontsize-10)
# #ax4[1,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax6[1,0].set_title('Sakonowyak River', fontsize=fontsize-3)
# plt.setp(ax6[1,0].get_xticklabels(), visible=False)

# Kuparuk
#df_kup.plot.area(ax=ax5[1,2], color=colors5, legend=False, sharex=True)
ax6[1,0].stackplot(df_kup.index, df_kup['0 - 10 m'], df_kup['10 - 20 m'], df_kup['20 - 30 m'], 
              df_kup['30 - 60 m'], labels=df_kup.columns, colors=colors6)
#plt.setp(ax5[1,2].get_yticklabels(), color='royalblue')
#ax5_12 = ax5[1,2].twinx()
# Get discharge 
#dis_kup = np.sum(river_frc.river_transport[:,12:14], axis=1)
#ax5_12.plot(river_frc.river_time[:40].values, dis_kup[:40], 
 #           color='green', linewidth=4)
#plt.setp(ax5_12.get_yticklabels(), color='green')
#ax4[1,2].legend(fontsize=fontsize-10)
#ax4[1,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[1,0].set_title('Kuparik River', fontsize=fontsize-3)
plt.setp(ax6[1,0].get_xticklabels(), visible=False)
#ax6[1,0].set_ylim([0,3e6])

# # Putuligayuk
# #df_put.plot.area(ax=ax5[2,0], color=colors5, legend=False, sharex=True)
# ax6[1,2].stackplot(df_put.index, df_put['0 - 10 m'], df_put['10 - 20 m'], df_put['20 - 30 m'], 
#               df_put['30 - 60 m'], labels=df_put.columns, colors=colors6)
# #plt.setp(ax6[1,2].get_yticklabels(), color='royalblue')
# #ax5_20 = ax5[1,2].twinx()
# #ax5_20.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,14].values, 
#  #           color='green', linewidth=4)
# #plt.setp(ax5_20.get_yticklabels(), color='green')
# #ax4[2,0].legend(fontsize=fontsize-10)
# #ax4[2,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax6[1,2].set_title('Putuligayuk River', fontsize=fontsize-3)
# plt.setp(ax6[1,2].get_xticklabels(), visible=False)

# Staines
#df_sta.plot.area(ax=ax5[2,1], color=colors5, legend=False, sharex=True)
ax6[1,1].stackplot(df_sta.index, df_sta['0 - 10 m'], df_sta['10 - 20 m'], df_sta['20 - 30 m'], 
              df_sta['30 - 60 m'], labels=df_sta.columns, colors=colors6)
#plt.setp(ax5[2,1].get_yticklabels(), color='royalblue')
#ax5_21 = ax5[2,1].twinx()
#ax5_21.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,15].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_21.get_yticklabels(), color='green')
#ax4[2,1].legend(fontsize=fontsize-10)
#ax4[2,1].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[1,1].set_title('Staines River', fontsize=fontsize-3)
plt.setp(ax6[1,1].get_xticklabels(), visible=False)
ax6[1,1].set_ylim([0,3e6])

# Canning
#df_can.plot.area(ax=ax5[2,2], color=colors5, legend=False, sharex=True)
ax6[1,2].stackplot(df_can.index, df_can['0 - 10 m'], df_can['10 - 20 m'], df_can['20 - 30 m'], 
              df_can['30 - 60 m'], labels=df_can.columns, colors=colors6)
#plt.setp(ax5[2,2].get_yticklabels(), color='royalblue')
#ax5_22 = ax5[2,2].twinx()
#ax5_22.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,16].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_22.get_yticklabels(), color='green')
#ax4[2,2].legend(fontsize=fontsize-10)
#ax4[2,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
plt.setp(ax6[1,2].get_xticklabels(), visible=True, fontsize=fontsize-3)
plt.xticks([pd.to_datetime('2020-07-01', format = '%Y-%m-%d'),  
            pd.to_datetime('2020-09-01', format = '%Y-%m-%d'),
            pd.to_datetime('2020-10-31', format = '%Y-%m-%d')])
ax6[1,2].set_title('Canning River', fontsize=fontsize-3)
#ax6[1,2].set_ylim([0,3e6])
#plt.setp(ax6[1,2].get_xticklabels(), visible=False)

# Katakturuk
#df_kat.plot.area(ax=ax5[3,0], color=colors5, legend=False)
ax6[1,3].stackplot(df_kat.index, df_kat['0 - 10 m'], df_kat['10 - 20 m'], df_kat['20 - 30 m'], 
              df_kat['30 - 60 m'], labels=df_kat.columns, colors=colors6)
#plt.setp(ax5[3,0].get_yticklabels(), color='royalblue')
#ax5_30 = ax5[3,0].twinx()
#ax5_30.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,17].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_30.get_yticklabels(), color='green')
#ax4[3,0].legend(fontsize=fontsize-10)
#ax4[3,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[1,3].set_title('Katakturuk River', fontsize=fontsize-3)
plt.setp(ax6[1,3].get_xticklabels(), visible=True, fontsize=fontsize-3)
#ax6[1,3].set_ylim([0,3e6])
plt.xticks([pd.to_datetime('2020-07-01', format = '%Y-%m-%d'), 
            pd.to_datetime('2020-09-01', format = '%Y-%m-%d'),
            pd.to_datetime('2020-10-31', format = '%Y-%m-%d')])
#plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# Hulahula - doing something weird with negative SSC...
#df_kat.plot.area(ax=ax5[3,0], color=colors5, legend=False)
ax6[2,0].stackplot(df_hul.index, df_hul['0 - 10 m'], df_hul['10 - 20 m'], df_hul['20 - 30 m'], 
              df_hul['30 - 60 m'], labels=df_hul.columns, colors=colors6)
#plt.setp(ax5[3,0].get_yticklabels(), color='royalblue')
#ax5_30 = ax5[3,0].twinx()
#ax5_30.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,17].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_30.get_yticklabels(), color='green')
#ax4[3,0].legend(fontsize=fontsize-10)
#ax4[3,0].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[2,0].set_title('Hulahula River', fontsize=fontsize-3)
plt.setp(ax6[2,0].get_xticklabels(), visible=True, fontsize=fontsize-3)
#ax6[2,0].set_ylim([0,3e6])
plt.xticks([pd.to_datetime('2020-07-01', format = '%Y-%m-%d'),  
            pd.to_datetime('2020-09-01', format = '%Y-%m-%d'),
            pd.to_datetime('2020-10-31', format = '%Y-%m-%d')])
#plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# Jago
#df_jag.plot.area(ax=ax5[3,1], color=colors5, legend=False)
ax6[2,1].stackplot(df_jag.index, df_jag['0 - 10 m'], df_jag['10 - 20 m'], df_jag['20 - 30 m'], 
              df_jag['30 - 60 m'], labels=df_jag.columns, colors=colors6)
#plt.setp(ax5[3,1].get_yticklabels(), color='royalblue')
#ax5_31 = ax5[3,1].twinx()
#ax5_31.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,19].values, 
 #           color='green', linewidth=4)
#plt.setp(ax5_31.get_yticklabels(), color='green')
#ax4[3,1].legend(fontsize=fontsize-10)
#ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
ax6[2,1].set_title('Jago River', fontsize=fontsize-3)
plt.setp(ax6[2,1].get_xticklabels(), visible=True, fontsize=fontsize-3)
#ax6[2,1].set_ylim([0,3e6])
#plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])
plt.xticks([pd.to_datetime('2020-07-01', format = '%Y-%m-%d'),  
            pd.to_datetime('2020-09-01', format = '%Y-%m-%d'),
            pd.to_datetime('2020-10-31', format = '%Y-%m-%d')])
#pd.to_datetime('2019-07-01', format = '%Y-%m-%d')

# # Siksik
# #df_sik.plot.area(ax=ax5[3,2], color=colors5, legend=False) #, xticks=['2019-07-01', '2019-07-15', '2019-08-01', '2019-08-15']
# ax6[3,0].stackplot(df_sik.index, df_sik['0 - 10 m'], df_sik['10 - 20 m'], df_sik['20 - 30 m'], 
#               df_sik['30 - 60 m'], labels=df_sik.columns, colors=colors5)
# #plt.setp(ax5[3,2].get_yticklabels(), color='royalblue')
# #ax5_32 = ax5[3,2].twinx()
# #ax5_32.plot(river_frc.river_time[:40].values, river_frc.river_transport[:40,20].values, 
#  #           color='green', linewidth=4)
# #plt.setp(ax5_32.get_yticklabels(), color='green')
# #ax4[3,2].legend(fontsize=fontsize-10)
# #ax4[3,2].set_ylabel('Sediment Mass (kg)', fontsize=fontsize-8)
# ax6[3,0].set_xlabel('Time', fontsize=fontsize+5)
# ax6[3,0].set_title('Siksik River', fontsize=fontsize-3)
# plt.setp(ax6[3,0].get_xticklabels(), visible=True, fontsize=fontsize-3)
# plt.xticks([pd.to_datetime('2019-07-01', format = '%Y-%m-%d'), 
#             pd.to_datetime('2019-07-25', format = '%Y-%m-%d'), 
#             pd.to_datetime('2019-08-15', format = '%Y-%m-%d')])
# #plt.xticks(['2019-07-01', '2019-07-25', '2019-08-15'])

# Set universal axes labels 
#fig4.supxlabel('Time', fontsize=fontsize, labelpad=2)
#fig4.supylabel('Sediment Mass (kg)', fontsize=fontsize, labelpad=0.2)

# Hide the other axes
ax6[2,2].axis('off')
ax6[2,3].axis('off')
#ax6[3,3].axis('off')

#fig6.text(0.5, 0.03, "Time", ha="center", va="center", fontsize=fontsize+5)
fig6.text(0.05, 0.55, "Sediment \nMass (kg)", ha="center", va="center", rotation=0, fontsize=fontsize+5) #, color='royalblue')
#fig6.text(0.97, 0.55, "River \nDischarge \n(m\u00b3/s)", ha="center", va="center", rotation=0, fontsize=fontsize+5, color='green')
fig6.text(0.5, 0.92, 'River Sediment Mass by Depth & Time (kg)', ha="center", va="center", rotation=0, fontsize=fontsize+5)

# Set universal legend 
# Put a legend for the rivers
#ax6[3,2].legend(fontsize=fontsize-4, loc='lower left', ncol=1, 
 #             labelspacing=0.1,  bbox_to_anchor=(1.17, 0.1))
ax6[2,3].legend(fontsize=fontsize+2, loc='center left', ncol=1, 
              labelspacing=0.1,  bbox_to_anchor=(0.25, -0.75))

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.22, wspace=0.35) #0.08

# Save the figure
# Aggregated
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_allrivs_2020_aggregated_dbsed0001_full_0004.png', transparent=True, bbox_inches='tight', pad_inches=0)
# Unaggregated
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/river_sed_kg_depth_allrivs_2020_unaggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)




# --------------------------------------------------------------------------------
# ----------------------------- Save the data to a NetCDF -------------------------
# --------------------------------------------------------------------------------
# Save to a netcdf
# Save these to a netcdf since the analysis takes so long?
kg_suspended_sed_from_rivers_in_depth_regions = xr.Dataset(
    data_vars=dict(
         kg_sed_0_10m_kal=(['ocean_time'], kg_sed_0_10m_kal_masked_trimmed_sum[:]),
         kg_sed_10_20m_kal=(['ocean_time'], kg_sed_10_20m_kal_masked_trimmed_sum[:]),
         kg_sed_20_30m_kal=(['ocean_time'], kg_sed_20_30m_kal_masked_trimmed_sum[:]),
         kg_sed_30_60m_kal=(['ocean_time'], kg_sed_30_60m_kal_masked_trimmed_sum[:]),
         kg_sed_0_10m_col=(['ocean_time'], kg_sed_0_10m_col_masked_trimmed_sum[:]),
         kg_sed_10_20m_col=(['ocean_time'], kg_sed_10_20m_col_masked_trimmed_sum[:]),
         kg_sed_20_30m_col=(['ocean_time'], kg_sed_20_30m_col_masked_trimmed_sum[:]),
         kg_sed_30_60m_col=(['ocean_time'], kg_sed_30_60m_col_masked_trimmed_sum[:]),
         kg_sed_0_10m_sag=(['ocean_time'], kg_sed_0_10m_sag_masked_trimmed_sum[:]),
         kg_sed_10_20m_sag=(['ocean_time'], kg_sed_10_20m_sag_masked_trimmed_sum[:]),
         kg_sed_20_30m_sag=(['ocean_time'], kg_sed_20_30m_sag_masked_trimmed_sum[:]),
         kg_sed_30_60m_sag=(['ocean_time'], kg_sed_30_60m_sag_masked_trimmed_sum[:]),
         kg_sed_0_10m_fis=(['ocean_time'], kg_sed_0_10m_fis_masked_trimmed_sum[:]),
         kg_sed_10_20m_fis=(['ocean_time'], kg_sed_10_20m_fis_masked_trimmed_sum[:]),
         kg_sed_20_30m_fis=(['ocean_time'], kg_sed_20_30m_fis_masked_trimmed_sum[:]),
         kg_sed_30_60m_fis=(['ocean_time'], kg_sed_30_60m_fis_masked_trimmed_sum[:]),
         kg_sed_0_10m_kup=(['ocean_time'], kg_sed_0_10m_kup_masked_trimmed_sum[:]),
         kg_sed_10_20m_kup=(['ocean_time'], kg_sed_10_20m_kup_masked_trimmed_sum[:]),
         kg_sed_20_30m_kup=(['ocean_time'], kg_sed_20_30m_kup_masked_trimmed_sum[:]),
         kg_sed_30_60m_kup=(['ocean_time'], kg_sed_30_60m_kup_masked_trimmed_sum[:]),
         kg_sed_0_10m_sta=(['ocean_time'], kg_sed_0_10m_sta_masked_trimmed_sum[:]),
         kg_sed_10_20m_sta=(['ocean_time'], kg_sed_10_20m_sta_masked_trimmed_sum[:]),
         kg_sed_20_30m_sta=(['ocean_time'], kg_sed_20_30m_sta_masked_trimmed_sum[:]),
         kg_sed_30_60m_sta=(['ocean_time'], kg_sed_30_60m_sta_masked_trimmed_sum[:]),
         kg_sed_0_10m_can=(['ocean_time'], kg_sed_0_10m_can_masked_trimmed_sum[:]),
         kg_sed_10_20m_can=(['ocean_time'], kg_sed_10_20m_can_masked_trimmed_sum[:]),
         kg_sed_20_30m_can=(['ocean_time'], kg_sed_20_30m_can_masked_trimmed_sum[:]),
         kg_sed_30_60m_can=(['ocean_time'], kg_sed_30_60m_can_masked_trimmed_sum[:]),
         kg_sed_0_10m_kat=(['ocean_time'], kg_sed_0_10m_kat_masked_trimmed_sum[:]),
         kg_sed_10_20m_kat=(['ocean_time'], kg_sed_10_20m_kat_masked_trimmed_sum[:]),
         kg_sed_20_30m_kat=(['ocean_time'], kg_sed_20_30m_kat_masked_trimmed_sum[:]),
         kg_sed_30_60m_kat=(['ocean_time'], kg_sed_30_60m_kat_masked_trimmed_sum[:]),
         kg_sed_0_10m_hul=(['ocean_time'], kg_sed_0_10m_hul_masked_trimmed_sum[:]),
         kg_sed_10_20m_hul=(['ocean_time'], kg_sed_10_20m_hul_masked_trimmed_sum[:]),
         kg_sed_20_30m_hul=(['ocean_time'], kg_sed_20_30m_hul_masked_trimmed_sum[:]),
         kg_sed_30_60m_hul=(['ocean_time'], kg_sed_30_60m_hul_masked_trimmed_sum[:]),
         kg_sed_0_10m_jag=(['ocean_time'], kg_sed_0_10m_jag_masked_trimmed_sum[:]),
         kg_sed_10_20m_jag=(['ocean_time'], kg_sed_10_20m_jag_masked_trimmed_sum[:]),
         kg_sed_20_30m_jag=(['ocean_time'], kg_sed_20_30m_jag_masked_trimmed_sum[:]),
         kg_sed_30_60m_jag=(['ocean_time'], kg_sed_30_60m_jag_masked_trimmed_sum[:])
    ),
    coords=dict(
        ocean_time=('ocean_time', time_steps)
    ),
    attrs=dict(description='Time-series ROMS output of suspended sediment mass from each river in each depth region')
)

# Save to a netcdf
# Aggregated 
#kg_suspended_sed_from_rivers_in_depth_regions.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/kg_suspended_sed_from_rivers_in_depth_regions_aggregated_dbsed0001.nc')
# Unaggregated 
#kg_suspended_sed_from_rivers_in_depth_regions.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/River_sed_depths/kg_suspended_sed_from_rivers_in_depth_regions_unaggregated_dbsed0001.nc')









# Need to....
# Loop  through output and get spatial time series of the different 
# river sediment classes
# kg/m3

# sort them to be by river -add Kukpuk and Kuparuk together 

# Depth-integrate - multiply by height, add together, multiply by dx dy
# kg per grid cell in horizontal

# multiply by mask to get region we trust

# sort by depth (and no nans! or that's fine but make sure math is good)

# sum over the regions to get total kg as function of time for 
# each river for each river/sediment class

# Make some plotssss















