############ Plot Box and Whisker Plots of Percent of ###################################
################ Riverine Sediment In Depths Over Time ######################
# The purpose of this script is to make box and whiskers plots showing the range of 
# percentages of riverine suspended sediments in different depths over all times. 
# The vision is to have a plot with four panels, the bottom being 0 - 10 m, then
# 10 - 20 m, 20 - 30 m, and 30 - 60 m. There will be a version with and without 
# Section 13 since that will be exlucded from the results but is good to have for now. 
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
#file_names = glob('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Aggregated/dbsed0001_10rivs/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
#file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_002_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
# -- Unaggregated --
# dbsed0002
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_003/Model_Output/dbsed0002/ocean_his_beaufort_rivers_13rivs_13seabed_unaggregated_dbsed0002_*.nc')
# 2020 dbsed0001 - full run
#file_names = glob('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Unaggregated/dbsed0001_10rivs_unaggregated/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc') 
file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_003_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc')

# Sort them to be in order
file_names2 = sorted(file_names)

# Check to see if this worked
print(file_names2[0], flush=True)
print(file_names2[1], flush=True)
print(file_names2[2], flush=True)
print(file_names2[-1], flush=True)
print('all files: ', file_names2, flush=True)

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




# # --------------------------------------------------------------------------------
# # --------- Plot 1: Maps of kg Sediment by Region --------------------------------
# # --------------------------------------------------------------------------------
# # Now that things are split up by region, plot to make sure it worked
# # Make a fake xy with the right resolution to be able to plot without the angle
# x_rho_flat = np.arange(0,750*len(grid.x_rho[0,:]),750)
# y_rho_flat = np.arange(0,600*len(grid.y_rho[:,0]),600)
# # Prep the data by  ultiplying by the mask and trimming
# # Trim 
# x_rho_flat_trimmed = x_rho_flat[c_west:-c_west]
# # Prep the data by  ultiplying by the mask and trimming
# # Multiply by mask
# h_masked = grid.h.values*grid.mask_rho.values*mask_rho_nan.nudge_mask_rho_nan
# # Trim 
# lon_rho_trimmed = grid.lon_rho[:,c_west:-c_west].values
# lat_rho_trimmed = grid.lat_rho[:,c_west:-c_west].values
# h_masked_trimmed = h_masked[:,c_west:-c_west]


# # Make the figure
# fig1, ax1 = plt.subplots(4, figsize=(22,21)) # (18,8) (26,12) (26,8) (26,10)

# # Set the colormaps
# cmap1 = cmocean.cm.turbid
# cmap2 = cmocean.cm.delta # ero depo

# # Set colorbar levels for bathymetry
# lev1 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# # Set colorbar levels for  ssc
# lev2 = np.arange(0,0.05,0.001) # kg

# # Kalikpik 0 - 10 m
# ax1[0].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
#                facecolor ='darkgray', alpha = 0.8)
# ax1[0].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
#                facecolor ='white', alpha = 0.8)
# cs1 = ax1[0].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
#                   kg_sed_0_10m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# # Plot bathymetry contours
# ax1[0].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# # Label the plot
# #ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
# plt.setp(ax1[0].get_xticklabels(), visible=False)
# #ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
# ax1[0].set_ylabel('Y (km)', fontsize=fontsize)

# # Kalikpik 10 - 20 m
# ax1[1].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
#                facecolor ='darkgray', alpha = 0.8)
# ax1[1].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
#                facecolor ='white', alpha = 0.8)
# cs2 = ax1[1].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
#                   kg_sed_10_20m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# # Plot bathymetry contours
# ax1[1].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# # Label the plot
# #ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
# plt.setp(ax1[1].get_xticklabels(), visible=False)
# #ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
# ax1[1].set_ylabel('Y (km)', fontsize=fontsize)

# # Kalikpik 20 - 30 m
# ax1[2].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
#                facecolor ='darkgray', alpha = 0.8)
# ax1[2].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
#                facecolor ='white', alpha = 0.8)
# cs3 = ax1[2].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
#                   kg_sed_20_30m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# # Plot bathymetry contours
# ax1[2].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# # Label the plot
# #ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
# plt.setp(ax1[2].get_xticklabels(), visible=False)
# #ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
# ax1[2].set_ylabel('Y (km)', fontsize=fontsize)

# # Kalikpik 30 - 60 m
# ax1[3].fill_between(x_rho_flat_trimmed/1000, 0, 65, 
#                facecolor ='darkgray', alpha = 0.8)
# ax1[3].fill_between(x_rho_flat_trimmed/1000, 65 ,120, 
#                facecolor ='white', alpha = 0.8)
# cs4 = ax1[3].contourf(x_rho_flat_trimmed/1000, y_rho_flat/1000,
#                   kg_sed_30_60m_kal_masked_trimmed[50,:,:], lev2, cmap=cmap1, extend='max')
# # Plot bathymetry contours
# ax1[3].contour(x_rho_flat_trimmed/1000, y_rho_flat/1000, h_masked_trimmed, lev1, colors='bisque')
# # Label the plot
# #ax7.set_title('Time-Averaged Surface Currents (m/s)', fontsize=fontsize, y=1.08)
# plt.setp(ax1[3].get_xticklabels(), visible=False)
# #ax8[0].set_xlabel('Longitude (degrees)', fontsize=fontsize)
# ax1[3].set_ylabel('Y (km)', fontsize=fontsize)

# cbar1 = plt.colorbar(cs4, orientation='vertical', ax=[ax1[0], ax1[1], ax1[2], ax1[3]]).set_label(label='Kalikpik Sediment (kg)', size=fontsize)



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

# ---- Percent Over Time ----
# *** START HERE ***
# Calculate the percentage that is locally resuspended 
# Make a function to calculate the percentage over time 
def get_percent_regions_over_time_by_river(kg_sed_0_10m_riv, kg_sed_10_20m_riv, kg_sed_20_30m_riv, kg_sed_30_60m_riv):
    """
    This function takes given arrays of riverine suspended sediment (kg) in each depth
    region over time and calculates the percentage in each region over time. 

    Inputs:
    - kg_sed_0_10m_riv: Time series of the riverine suspended sediment (kg) from a given
                        river that is in 0 - 10 m depth
    - kg_sed_10_20m_riv: Time series of the riverine suspended sediment (kg) from a given
                        river that is in 10 - 20 m depth
    - kg_sed_20_30m_riv: Time series of the riverine suspended sediment (kg) from a given
                        river that is in 20 - 30 m depth
    - kg_sed_30_60m_riv: Time series of the riverine suspended sediment (kg) from a given
                        river that is in 30 - 60 m depth
    

    
    Outputs:
    - percent_0_10m: Time series of the percentage of riverine suspended sediment 
                      that is suspended in 0 - 10 m water depths over time, relative
                      to the total riverine suspended sediment from that river at that time
    - percent_10_20m: Time series of the percentage of riverine suspended sediment 
                      that is suspended in 10 - 20 m water depths over time, relative
                      to the total riverine suspended sediment from that river at that time
    - percent_20_30m: Time series of the percentage of riverine suspended sediment 
                      that is suspended in 20 - 30 m water depths over time, relative
                      to the total riverine suspended sediment from that river at that time
    - percent_30_60m: Time series of the percentage of riverine suspended sediment 
                      that is suspended in 30 - 60 m water depths over time, relative
                      to the total riverine suspended sediment from that river at that time
    """

    # Sum up all of the suspended sediment from this river in all depths over time
    tot_ssc_from_river = kg_sed_0_10m_riv + kg_sed_10_20m_riv + kg_sed_20_30m_riv + kg_sed_30_60m_riv

    # Calculate the percent that is in each depth range
    # 0 - 10 m
    percent_0_10m = (kg_sed_0_10m_riv/tot_ssc_from_river)*100
    # Check this 
    print('kg_sed_0_10m_riv[24]: ', kg_sed_0_10m_riv[24])
    print('tot_ssc_from_river[24]: ', tot_ssc_from_river[24])
    print('percent_0_10m[24]: ', percent_0_10m[24])
    # 10 - 20 m
    percent_10_20m = (kg_sed_10_20m_riv/tot_ssc_from_river)*100
    # 20 - 30 m
    percent_20_30m = (kg_sed_20_30m_riv/tot_ssc_from_river)*100
    # 30 - 60 m
    percent_30_60m = (kg_sed_30_60m_riv/tot_ssc_from_river)*100

    # Return this value...
    return(percent_0_10m, percent_10_20m, percent_20_30m, percent_30_60m)


# Call the function for each river 
# Kalikpik
percent_0_10m_kal, percent_10_20m_kal, percent_20_30m_kal, percent_30_60m_kal = get_percent_regions_over_time_by_river(kg_sed_0_10m_kal_masked_trimmed_sum, kg_sed_10_20m_kal_masked_trimmed_sum, kg_sed_20_30m_kal_masked_trimmed_sum, kg_sed_30_60m_kal_masked_trimmed_sum)
# Colville
percent_0_10m_col, percent_10_20m_col, percent_20_30m_col, percent_30_60m_col = get_percent_regions_over_time_by_river(kg_sed_0_10m_col_masked_trimmed_sum, kg_sed_10_20m_col_masked_trimmed_sum, kg_sed_20_30m_col_masked_trimmed_sum, kg_sed_30_60m_col_masked_trimmed_sum)
# Sagavanirktok
percent_0_10m_sag, percent_10_20m_sag, percent_20_30m_sag, percent_30_60m_sag = get_percent_regions_over_time_by_river(kg_sed_0_10m_sag_masked_trimmed_sum, kg_sed_10_20m_sag_masked_trimmed_sum, kg_sed_20_30m_sag_masked_trimmed_sum, kg_sed_30_60m_sag_masked_trimmed_sum)
# Fish Creek
percent_0_10m_fis, percent_10_20m_fis, percent_20_30m_fis, percent_30_60m_fis = get_percent_regions_over_time_by_river(kg_sed_0_10m_fis_masked_trimmed_sum, kg_sed_10_20m_fis_masked_trimmed_sum, kg_sed_20_30m_fis_masked_trimmed_sum, kg_sed_30_60m_fis_masked_trimmed_sum)
# Sakonowyak
#percent_0_10m_sak, percent_10_20m_sak, percent_20_30m_sak, percent_30_60m_sak = get_percent_regions_over_time_by_river(kg_sed_0_10m_sak_masked_trimmed_sum, kg_sed_10_20m_sak_masked_trimmed_sum, kg_sed_20_30m_sak_masked_trimmed_sum, kg_sed_30_60m_sak_masked_trimmed_sum)
# Kuparuk
percent_0_10m_kup, percent_10_20m_kup, percent_20_30m_kup, percent_30_60m_kup = get_percent_regions_over_time_by_river(kg_sed_0_10m_kup_masked_trimmed_sum, kg_sed_10_20m_kup_masked_trimmed_sum, kg_sed_20_30m_kup_masked_trimmed_sum, kg_sed_30_60m_kup_masked_trimmed_sum)
# Putuligayuk
#percent_0_10m_put, percent_10_20m_put, percent_20_30m_put, percent_30_60m_put = get_percent_regions_over_time_by_river(kg_sed_0_10m_put_masked_trimmed_sum, kg_sed_10_20m_put_masked_trimmed_sum, kg_sed_20_30m_put_masked_trimmed_sum, kg_sed_30_60m_put_masked_trimmed_sum)
# Staines
percent_0_10m_sta, percent_10_20m_sta, percent_20_30m_sta, percent_30_60m_sta = get_percent_regions_over_time_by_river(kg_sed_0_10m_sta_masked_trimmed_sum, kg_sed_10_20m_sta_masked_trimmed_sum, kg_sed_20_30m_sta_masked_trimmed_sum, kg_sed_30_60m_sta_masked_trimmed_sum)
# Canning
percent_0_10m_can, percent_10_20m_can, percent_20_30m_can, percent_30_60m_can = get_percent_regions_over_time_by_river(kg_sed_0_10m_can_masked_trimmed_sum, kg_sed_10_20m_can_masked_trimmed_sum, kg_sed_20_30m_can_masked_trimmed_sum, kg_sed_30_60m_can_masked_trimmed_sum)
# Katakturuk
percent_0_10m_kat, percent_10_20m_kat, percent_20_30m_kat, percent_30_60m_kat = get_percent_regions_over_time_by_river(kg_sed_0_10m_kat_masked_trimmed_sum, kg_sed_10_20m_kat_masked_trimmed_sum, kg_sed_20_30m_kat_masked_trimmed_sum, kg_sed_30_60m_kat_masked_trimmed_sum)
# Hulahula
percent_0_10m_hul, percent_10_20m_hul, percent_20_30m_hul, percent_30_60m_hul = get_percent_regions_over_time_by_river(kg_sed_0_10m_hul_masked_trimmed_sum, kg_sed_10_20m_hul_masked_trimmed_sum, kg_sed_20_30m_hul_masked_trimmed_sum, kg_sed_30_60m_hul_masked_trimmed_sum)
# Jago
percent_0_10m_jag, percent_10_20m_jag, percent_20_30m_jag, percent_30_60m_jag = get_percent_regions_over_time_by_river(kg_sed_0_10m_jag_masked_trimmed_sum, kg_sed_10_20m_jag_masked_trimmed_sum, kg_sed_20_30m_jag_masked_trimmed_sum, kg_sed_30_60m_jag_masked_trimmed_sum)


# Save to a netcdf
# Save these to a netcdf since the analysis takes so long?
percent_regions_over_time_byriver = xr.Dataset(
    data_vars=dict(
        percent_0_10m_kal=(['ocean_time'], percent_0_10m_kal),
        percent_10_20m_kal=(['ocean_time'], percent_10_20m_kal),
        percent_20_30m_kal=(['ocean_time'], percent_20_30m_kal),
        percent_30_60m_kal=(['ocean_time'], percent_30_60m_kal),
        percent_0_10m_col=(['ocean_time'], percent_0_10m_col),
        percent_10_20m_col=(['ocean_time'], percent_10_20m_col),
        percent_20_30m_col=(['ocean_time'], percent_20_30m_col),
        percent_30_60m_col=(['ocean_time'], percent_30_60m_col),
        percent_0_10m_sag=(['ocean_time'], percent_0_10m_sag),
        percent_10_20m_sag=(['ocean_time'], percent_10_20m_sag),
        percent_20_30m_sag=(['ocean_time'], percent_20_30m_sag),
        percent_30_60m_sag=(['ocean_time'], percent_30_60m_sag),
        percent_0_10m_fis=(['ocean_time'], percent_0_10m_fis),
        percent_10_20m_fis=(['ocean_time'], percent_10_20m_fis),
        percent_20_30m_fis=(['ocean_time'], percent_20_30m_fis),
        percent_30_60m_fis=(['ocean_time'], percent_30_60m_fis),
        percent_0_10m_kup=(['ocean_time'], percent_0_10m_kup),
        percent_10_20m_kup=(['ocean_time'], percent_10_20m_kup),
        percent_20_30m_kup=(['ocean_time'], percent_20_30m_kup),
        percent_30_60m_kup=(['ocean_time'], percent_30_60m_kup),
        percent_0_10m_sta=(['ocean_time'], percent_0_10m_sta),
        percent_10_20m_sta=(['ocean_time'], percent_10_20m_sta),
        percent_20_30m_sta=(['ocean_time'], percent_20_30m_sta),
        percent_30_60m_sta=(['ocean_time'], percent_30_60m_sta),
        percent_0_10m_can=(['ocean_time'], percent_0_10m_can),
        percent_10_20m_can=(['ocean_time'], percent_10_20m_can),
        percent_20_30m_can=(['ocean_time'], percent_20_30m_can),
        percent_30_60m_can=(['ocean_time'], percent_30_60m_can),
        percent_0_10m_kat=(['ocean_time'], percent_0_10m_kat),
        percent_10_20m_kat=(['ocean_time'], percent_10_20m_kat),
        percent_20_30m_kat=(['ocean_time'], percent_20_30m_kat),
        percent_30_60m_kat=(['ocean_time'], percent_30_60m_kat),
        percent_0_10m_hul=(['ocean_time'], percent_0_10m_hul),
        percent_10_20m_hul=(['ocean_time'], percent_10_20m_hul),
        percent_20_30m_hul=(['ocean_time'], percent_20_30m_hul),
        percent_30_60m_hul=(['ocean_time'], percent_30_60m_hul),
        percent_0_10m_jag=(['ocean_time'], percent_0_10m_jag),
        percent_10_20m_jag=(['ocean_time'], percent_10_20m_jag),
        percent_20_30m_jag=(['ocean_time'], percent_20_30m_jag),
        percent_30_60m_jag=(['ocean_time'], percent_30_60m_jag)
    ),
    coords=dict(
        ocean_time=('ocean_time', time_steps)
    ),
    attrs=dict(description='Time-series ROMS output of percent of riverine suspended sediment in each region by river')
)

# Save to a netcdf
# Aggregated 
#percent_regions_over_time_byriver.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_river_suspended_sed_regions/percent_river_suspended_sed_over_time_in_regions_aggregated.nc')
# Unaggregated 
percent_regions_over_time_byriver.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_river_suspended_sed_regions/percent_river_suspended_sed_over_time_in_regions_unaggregated.nc')



# Make a list to plot?


# Try to plot?



# Make a box and whiskers plot of percentages

