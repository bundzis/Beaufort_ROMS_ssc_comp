############ Plot Box and Whisker Plots of Percent of ###################################
##### Suspended Sediment that is Locally Suspended Over Time ############################
# The purpose of this script is to make plots that have one box and whisker plot for 
# each shelf section that has the range of percentages of locally resuspended 
# sediment in that section over time. 
#
# Notes:
# - This leaves out the rivers that are no longer in the 2020 model
#   runs that use Blaskey river data 
#########################################################################################


# Kalikpik
# Colville
# Sagavanirktok
# Fish Creek
# Sakonowyak, removed because not in Blaskey
# Kuparuk
# Putuligayuk, removed because not in Blaskey
# Staines
# Canning 
# Katakturuk
# Hulahula
# Jago
# Siksik, removed because not in Blaskey
# Section 1
# Section 2
# Section 3
# Section 4
# Section 5
# Section 6
# Section 7
# Section 8 
# Section 9
# Section 10 
# Section 11
# Section 12
# Section 13

# USE BELOW ORDER IN FUTURE
# Kalikpik
# Fish Creek
# Colville
# Sakonowyak
# Kuparuk
# Putuligayuk
# Sagavanirktok
# Staines
# Canning 
# Katakturuk
# Hulahula
# Jago
# Siksik

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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as mcolors

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
Nbed_len = 11

# Multiply by masks to make land appear 
# Make it so land will appear
temp_mask = grid.mask_rho.copy()
temp_mask = np.where(temp_mask==0, np.nan, temp_mask)


# Load in the rho masks 
mask_rho_nan = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_ones_nans.nc') # UPDATE PATH
mask_rho_zeros = xr.open_dataset('/projects/brun1463/ROMS/Kakak3_Alpine/Scripts_2/Analysis/Nudge_masks/nudge_mask_rho_zeros_ones.nc')
#mask_rho_nan = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_ones_nans.nc')
#mask_rho_zeros = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Kaktovik_Alaska_2019/Code/Nudge_masks/nudge_mask_rho_zeros_ones.nc')


# Load in the river forcing file 
# -- Aggregated --
#river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/Rivers/river_forcing_file_beaufort_shelf_13rivs_13seabed_radr_data_001.nc')
# (2020)
river_frc = xr.open_dataset('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Include/river_forcing_file_beaufort_shelf_10rivs_13seabed_blaskey_data_sagDSS3_rating_001.nc')
# -- Unaggregated (it is the same for now) --
#river_frc = xr.open_dataset('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Input/Rivers/river_forcing_file_beaufort_shelf_13rivs_13seabed_radr_data_003.nc')


# -------------------------------------------------------------------------------------
# ----------------------- Define a Bunch o Functions ----------------------------
# -------------------------------------------------------------------------------------

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


# Depth-integrated 
# Make a function to calculate the depth-integrated SSC and 
# depth-averaged SSC for a specific sediment class
def calc_depth_int_avg_ssc_1sed(filename, sediment_class):
    """
    The purpose of this function is to take a given model output file, load 
    in the output, and caluclate the depth-integrated and depth-averaegd 
    suspended sediment concentrations (SSC) for a specified sediment class

    Parameters
    ----------
    filename : The name/path of the model output file.
    sediment_class : The name of the sediment class of interest; ex: 'mud_01'

    Returns
    -------
    depth_int_ssc_1sed: Spatail time series of depth-integrated SSC for 
    specified sediment class (kg/m2)
    depth_avg_ssc_1sed: Spatial time series of depth-averaged ssc for a 
    specified sediment class (kg/m3)

    """
    
    # Load in the model output
    model_output = xr.open_dataset(filename)
    
    # Pull out the sediment class of interest
    ssc_1sed_tmp = model_output[sediment_class]
    
    # To collapse to horizontal, multiply each layer by its
    # thickness
    # Calculate the time-varying thickness of the cells
    dz = abs(model_output.z_w[:,:-1,:,:].values - model_output.z_w[:,1:,:,:].values)
    
    # Calculate depth-integrated ssc
    depth_int_ssc_1sed = (((ssc_1sed_tmp*dz)).sum(dim='s_rho'))
    
    # Divide by bathymetry to get depth-averaged SSC (kg/m3)
    depth_avg_ssc_1sed = depth_int_ssc_1sed/model_output.bath[:,:,:].values
    
    # Return the depth-integrated u flux for all sediment classes
    return(depth_int_ssc_1sed, depth_avg_ssc_1sed)


# Make functions to help with masking 
# Make masks to isolate each region, starting with breaking things up by depth 
# Make a function to mask the data
def masked_array(data, threshold):
    """
    This function takes an array and masks all values that are less
    than a certain given threshold. The functions returns 1 for areas that meet 
    the condition and 0 for areas that don't. So areas where the array is less
    than the threshold get returned as 1 and areas greater than the threshold
    are returned as 0. This function maintains the shape of the array.
    
    """
    return (data <= threshold).astype(int)

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

# Make a function to mask the data - higher
def masked_array_higher(data, threshold):
    """
    This function takes an array and masks all values that are less
    than a certain given threshold. The functions returns 1 for areas that meet 
    the condition and 0 for areas that don't. So areas where the array is less
    than the threshold get returned as 1 and areas greater than the threshold
    are returned as 0. This function maintains the shape of the array.
    
    """
    return (data >= threshold).astype(int)


# Make a function that takes a given color and makes a color out of it
def saturation_colormap(color, N=256):
    """
    This function creates a colormap with different saturation levels of a 
    single sepcified color

    Parameters
    ----------
    color : Desired color for colormap
    N : number of colors to include in the map
        DESCRIPTION. The default is 256.

    Returns
    -------
    None.

    """
    
    # Convert color to RGB if it's a hex code
    if isinstance(color, str):
        color = plt.cm.colors.to_rgb(color)

    # Create a list of colors with varying saturation
    colors = []
    for sat in np.linspace(0, 1, N):
        colors.append((color[0], color[1], color[2], sat))

    return ListedColormap(colors)


# -------------------------------------------------------------------------------------
# ----------------------- Prep file names and locations ----------------------------
# -------------------------------------------------------------------------------------
# Now 
# Loop through model output and call the function
# First, get all the file names
# -- Aggregated --
# Separate rivers and seabeds - dbsed0003
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Outputs/dbsed0004/ocean_his_beaufort_rivers_13rivs_13seabed_dbsed0004_*.nc')
# Separate rivers and seabeds - dbsed0003
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_002/Model_Outputs/dbsed0009/ocean_his_beaufort_rivers_13rivs_13seabed_dbsed0009_*.nc')
# dbsed0001 2020 - full run **this path may beed to be updated to go to petalibrary** 
file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_002_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
# -- Unaggregated Mud --
#file_names = glob('/Users/brun1463/Desktop/Research_Lab/Beaufort_Shelf_Rivers_proj_003/Model_Output/dbsed0002/ocean_his_beaufort_rivers_13rivs_13seabed_unaggregated_dbsed0002_*.nc')
# dbsed0001 2020  **this path may need to be updated to go to petalibrary**
#file_names = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_003_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc')

# Sort them to be in order
file_names2 = sorted(file_names)

print('file names: ', file_names2)

# Check to see if this worked
# =============================================================================
# print(file_names2[0], flush=True)
# print(file_names2[1], flush=True)
# print(file_names2[2], flush=True)
# print(file_names2[-1], flush=True)
# =============================================================================

# Pull out the number of files
num_files = len(file_names2)

# Pull out the length of time of the full run, the time steps, 
# and the length of time of each output file
full_time_len, time_steps, time_lengths = get_model_time(file_names2, num_files)


# -------------------------------------------------------------------------------------
# --------------------- Process Output: Depth-Integrated SSC --------------------------
# -------------------------------------------------------------------------------------
# Make a function that does everything that is done below for a given sediment class
def get_timeseries_depth_int_avg_ssc_1sed(file_names, time_lengths, full_time_len, eta_rho_len, xi_rho_len, sed_class_ssc):
    """
    The purpose of this function is to get out two time series of the kg of sediment 
    per layer in the seabed and the frac of sediment per layer in the seabed for 
    a user-specified sediment class.
    

    Parameters
    ----------
    file_names : list of path and names of model outputs
    time_lengths : List of lengths of time in each output file 
    full_time_len : Lenght of time of all model output
    eta_rho_len : Length of eta_rho points
    xi_rho_len : Length of xi_rho points
    sed_class_ssc : Specified sediment class ssc, ex: 'mud_01'

    Returns
    -------
    None.

    """
    
    # Make empty arrays to hold the time series
    depth_int_ssc_1sed = np.empty((full_time_len, eta_rho_len, xi_rho_len))
    depth_avg_ssc_1sed = np.empty((full_time_len, eta_rho_len, xi_rho_len))
    
    # Print the sediment class
    print(sed_class_ssc)
    
    #
    # Set a time step to track which time step the loop is on
    time_step = 0

    # Loop through the model output
    for j in range(num_files):
    #for j in range(1):

        #print('j: ', j)
        
        # Prep time for saving to arrays
        #print('time_step: ', time_step)
        #print('time_step + time_lengths[j]: ', time_step+time_lengths[j])
        start = int(time_step)
        end = int(time_step+time_lengths[j])
        
        # Call the function to process the output - mass in seabed
        depth_int_ssc_1sed[start:end,:,:], depth_avg_ssc_1sed[start:end,:,:] = calc_depth_int_avg_ssc_1sed(file_names[j], sed_class_ssc)
        
        # Update the base time_step
        time_step = time_step + time_lengths[j]
    
    # Return these arrays 
    return(depth_int_ssc_1sed, depth_avg_ssc_1sed)
    
    
# Call the above function for each class to get the arrays
# Kalikpik
print('Kalikpik', flush=True)
depth_int_ssc_kal, depth_avg_ssc_kal = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_15')
# Colville
print('Colville', flush=True)
depth_int_ssc_col, depth_avg_ssc_col = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_16')
# Sagavanirktok
print('Sagavanirktok', flush=True)
depth_int_ssc_sag, depth_avg_ssc_sag = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_17')
# Fish Creek
print('Fish Creek', flush=True)
depth_int_ssc_fis, depth_avg_ssc_fis = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_18')
# Sakonowyak
#depth_int_ssc_sak, depth_avg_ssc_sak = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_19')
# Kuparik
print('Kuparik', flush=True)
depth_int_ssc_kup, depth_avg_ssc_kup = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_19') # mud_20
# Putuligayuk
#depth_int_ssc_put, depth_avg_ssc_put = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_21') # mud_21
# Staines
print('Staines', flush=True)
depth_int_ssc_sta, depth_avg_ssc_sta = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_20') # mud_22
# Canning 
print('Canning', flush=True)
depth_int_ssc_can, depth_avg_ssc_can = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_21') # mud_23
# Katakturuk
print('Katakturuk', flush=True)
depth_int_ssc_kat, depth_avg_ssc_kat = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_22') # mud_24
# Hulahula
print('Hulahula', flush=True)
depth_int_ssc_hul, depth_avg_ssc_hul = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_23') # mud_25
# Jago
print('Jago', flush=True)
depth_int_ssc_jag, depth_avg_ssc_jag = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len,eta_rho_len, xi_rho_len, 'mud_24') # mud_26
# Siksik
#depth_int_ssc_sik, depth_avg_ssc_sik = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_27')
# Section 1
print('Section 1', flush=True)
depth_int_ssc_sec1, depth_avg_ssc_sec1 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_02')
# Section 2
print('Section 2', flush=True)
depth_int_ssc_sec2, depth_avg_ssc_sec2 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_03')
# Section 3
print('Section 3', flush=True)
depth_int_ssc_sec3, depth_avg_ssc_sec3 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_04')
# Section 4
print('Section 4', flush=True)
depth_int_ssc_sec4, depth_avg_ssc_sec4 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_05')
# Section 5
print('Section 5', flush=True)
depth_int_ssc_sec5, depth_avg_ssc_sec5 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_06')
# Section 6
print('Section 6', flush=True)
depth_int_ssc_sec6, depth_avg_ssc_sec6 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_07')
# Section 7
print('Section 7', flush=True)
depth_int_ssc_sec7, depth_avg_ssc_sec7 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_08')
# Section 8
print('Section 8', flush=True)
depth_int_ssc_sec8, depth_avg_ssc_sec8 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_09')
# Section 9 
print('Section 9', flush=True)
depth_int_ssc_sec9, depth_avg_ssc_sec9 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_10')
# Section 10
print('Section 10', flush=True)
depth_int_ssc_sec10, depth_avg_ssc_sec10 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_11')
# Section 11
print('Section 11', flush=True)
depth_int_ssc_sec11, depth_avg_ssc_sec11 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_12')
# Section 12
print('Section 12', flush=True)
depth_int_ssc_sec12, depth_avg_ssc_sec12 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_13')
# Section 13
print('Section 13', flush=True)
depth_int_ssc_sec13, depth_avg_ssc_sec13 = get_timeseries_depth_int_avg_ssc_1sed(file_names2, time_lengths, full_time_len, eta_rho_len, xi_rho_len, 'mud_14')



# -------------------------------------------------------------------------------------
# --------------------- Process Output: Sort into Sections --------------------------
# -------------------------------------------------------------------------------------
# Start with a few sections for now to see if this is worth sorting that way

# Then need to sort into seabed sections (this is gonna be a LOT of variables...)
# so that we can look by section at the values 

# Make the masks for each section
# Make the masks to partition the seabed into different regions 

# Call the function to make the mask
# Leave in the region that is nudged/past the slope since we will need 
# to put sediment there regardless

# 0 - 10 m 
h_masked1 = grid.h.copy()
mask_0_10m_rho = masked_array_lowhigh_2dloop(h_masked1, 0, 10)

# 10 - 30 m
h_masked2 = grid.h.copy()
mask_10_30m_rho = masked_array_lowhigh_2dloop(h_masked2, 10, 30)

# 30 - 60 m
h_masked3 = grid.h.copy()
mask_30_60m_rho = masked_array_lowhigh_2dloop(h_masked3, 30, 60)

# > 60 m 
h_masked4 = grid.h.copy()
mask_60_end_rho = masked_array_lowhigh_2dloop(h_masked4, 60, 2200)

# Partition the seabed into the different regions 

# Make the nearshore sections 
mask_0_10m_rho_plot2 = mask_0_10m_rho.copy()
idx_ones_0_10m2 = np.where(mask_0_10m_rho_plot2 == 1)
# Set everywhere it is 1 equal to 2 
#mask_0_10m_rho_plot2[idx_ones_0_10m2] = 2
# Make an empty list to hold the indices for center
idx_ones_0_10m_far_left_etas = []
idx_ones_0_10m_far_left_xis = []
idx_ones_0_10m_center_left_etas = []
idx_ones_0_10m_center_left_xis = []
idx_ones_0_10m_center_right_etas = []
idx_ones_0_10m_center_right_xis = []
idx_ones_0_10m_far_right_etas = []
idx_ones_0_10m_far_right_xis = []
# Loop through these indices and find where they are in certain ranges
for i in range(len(idx_ones_0_10m2[0])):
    # Pull out the eta and the xi
    eta_tmp = idx_ones_0_10m2[0][i]
    xi_tmp = idx_ones_0_10m2[1][i]
    # Check if this is in the far left range
    if xi_tmp < 166:
        idx_ones_0_10m_far_left_etas.append(eta_tmp)
        idx_ones_0_10m_far_left_xis.append(xi_tmp)
    # Check if this is in the center left range
    elif 166 <= xi_tmp < 300:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_0_10m_center_left_etas.append(eta_tmp)
        idx_ones_0_10m_center_left_xis.append(xi_tmp)
    # Check if this is in the center right range
    elif 300 <= xi_tmp < 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_0_10m_center_right_etas.append(eta_tmp)
        idx_ones_0_10m_center_right_xis.append(xi_tmp)
        # Check if this is in the far right range
    elif xi_tmp >= 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_0_10m_far_right_etas.append(eta_tmp)
        idx_ones_0_10m_far_right_xis.append(xi_tmp)

# Make this 3 for center and 4 for right
mask_0_10m_rho_plot2[idx_ones_0_10m_far_left_etas[:],idx_ones_0_10m_far_left_xis[:]] = 2
mask_0_10m_rho_plot2[idx_ones_0_10m_center_left_etas[:],idx_ones_0_10m_center_left_xis[:]] = 3
mask_0_10m_rho_plot2[idx_ones_0_10m_center_right_etas[:],idx_ones_0_10m_center_right_xis[:]] = 4
mask_0_10m_rho_plot2[idx_ones_0_10m_far_right_etas[:],idx_ones_0_10m_far_right_xis[:]] = 5



# Make the middle sections 
mask_10_30m_rho_plot2 = mask_10_30m_rho.copy()
idx_ones_10_30m2 = np.where(mask_10_30m_rho_plot2 == 1)
# Set everything in here equal to 5 
#mask_10_30m_rho_plot2[idx_ones_10_30m2] = 5
# Make an empty list to hold the indices for center
idx_ones_10_30m_far_left_etas = []
idx_ones_10_30m_far_left_xis = []
idx_ones_10_30m_center_left_etas = []
idx_ones_10_30m_center_left_xis = []
idx_ones_10_30m_center_right_etas = []
idx_ones_10_30m_center_right_xis = []
idx_ones_10_30m_far_right_etas = []
idx_ones_10_30m_far_right_xis = []
# Loop through these indices and find where they are in certain ranges
for i in range(len(idx_ones_10_30m2[0])):
    # Pull out the eta and the xi
    eta_tmp = idx_ones_10_30m2[0][i]
    xi_tmp = idx_ones_10_30m2[1][i]
    # Check if this is in the far left range
    if xi_tmp < 166:
        idx_ones_10_30m_far_left_etas.append(eta_tmp)
        idx_ones_10_30m_far_left_xis.append(xi_tmp)
    # Check if this is in the center left range
    if 166 <= xi_tmp < 300:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_10_30m_center_left_etas.append(eta_tmp)
        idx_ones_10_30m_center_left_xis.append(xi_tmp)
    # Check if this is in the center right range
    if 300 <= xi_tmp < 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_10_30m_center_right_etas.append(eta_tmp)
        idx_ones_10_30m_center_right_xis.append(xi_tmp)
    # Check if this is in the far right range
    if xi_tmp >= 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_10_30m_far_right_etas.append(eta_tmp)
        idx_ones_10_30m_far_right_xis.append(xi_tmp)
# Make this 6 and 7 for the center and right 
mask_10_30m_rho_plot2[idx_ones_10_30m_far_left_etas[:],idx_ones_10_30m_far_left_xis[:]] = 6
mask_10_30m_rho_plot2[idx_ones_10_30m_center_left_etas[:],idx_ones_10_30m_center_left_xis[:]] = 7
mask_10_30m_rho_plot2[idx_ones_10_30m_center_right_etas[:],idx_ones_10_30m_center_right_xis[:]] = 8
mask_10_30m_rho_plot2[idx_ones_10_30m_far_right_etas[:],idx_ones_10_30m_far_right_xis[:]] = 9



# Make the outer sections
mask_30_60m_rho_plot2 = mask_30_60m_rho.copy()
idx_ones_30_60m2 = np.where(mask_30_60m_rho_plot2 == 1)
# Set everything in here equal to 8
#mask_30_60m_rho_plot2[idx_ones_30_60m2] = 8
# Make an empty list to hold the indices for center
idx_ones_30_60m_far_left_etas = []
idx_ones_30_60m_far_left_xis = []
idx_ones_30_60m_center_left_etas = []
idx_ones_30_60m_center_left_xis = []
idx_ones_30_60m_center_right_etas = []
idx_ones_30_60m_center_right_xis = []
idx_ones_30_60m_far_right_etas = []
idx_ones_30_60m_far_right_xis = []
# Loop through these indices and find where they are in certain ranges
for i in range(len(idx_ones_30_60m2[0])):
    # Pull out the eta and the xi
    eta_tmp = idx_ones_30_60m2[0][i]
    xi_tmp = idx_ones_30_60m2[1][i]
    # Check if this is in the far left range
    if xi_tmp < 166:
        idx_ones_30_60m_far_left_etas.append(eta_tmp)
        idx_ones_30_60m_far_left_xis.append(xi_tmp)
    # Check if this is in the center left range
    if 166 <= xi_tmp < 300:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_30_60m_center_left_etas.append(eta_tmp)
        idx_ones_30_60m_center_left_xis.append(xi_tmp)
    # Check if this is in the center right range
    if 300 <= xi_tmp < 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_30_60m_center_right_etas.append(eta_tmp)
        idx_ones_30_60m_center_right_xis.append(xi_tmp)
    # Check if this is in the far right range
    if xi_tmp >= 446:
        #print('xi_tmp: ', xi_tmp)
        #input('press enter to continue...')
        idx_ones_30_60m_far_right_etas.append(eta_tmp)
        idx_ones_30_60m_far_right_xis.append(xi_tmp)
# Make this 6 and 7 for the center and right 
mask_30_60m_rho_plot2[idx_ones_30_60m_far_left_etas[:],idx_ones_30_60m_far_left_xis[:]] = 10
mask_30_60m_rho_plot2[idx_ones_30_60m_center_left_etas[:],idx_ones_30_60m_center_left_xis[:]] = 11
mask_30_60m_rho_plot2[idx_ones_30_60m_center_right_etas[:],idx_ones_30_60m_center_right_xis[:]] = 12
mask_30_60m_rho_plot2[idx_ones_30_60m_far_right_etas[:],idx_ones_30_60m_far_right_xis[:]] = 13



# Make the outest section
mask_60_end_rho_plot2 = mask_60_end_rho.copy()
idx_ones_60_end2 = np.where(mask_60_end_rho_plot2 == 1)
# Set everything in here equal to 8
mask_60_end_rho_plot2[idx_ones_60_end2] = 14

# Add in the nudged sections to the last group = 14
# Set the number of cells in the sponge on each open boundary
c_west = 36
c_north = 45
c_east = 36
# Make these regions 14
mask_0_10m_rho_plot2[:,:c_west] = 14
mask_0_10m_rho_plot2[:,-c_west:] = 14
mask_10_30m_rho_plot2[:,:c_west] = 14
mask_10_30m_rho_plot2[:,-c_west:] = 14
mask_30_60m_rho_plot2[:,:c_west] = 14
mask_30_60m_rho_plot2[:,-c_west:] = 14
mask_60_end_rho_plot2[:,:c_west] = 14
mask_60_end_rho_plot2[:,-c_west:] = 14


# Assuming the above worked, use if statements and such to separate out the different regions
# and multiply them by mud02_pcnt to section this out

# Save the section arrays multiplied by the rho mask
mask_0_10m_rho_plot = mask_0_10m_rho_plot2*grid.mask_rho.values
mask_10_30m_rho_plot = mask_10_30m_rho_plot2*grid.mask_rho.values
mask_30_60m_rho_plot = mask_30_60m_rho_plot2*grid.mask_rho.values
mask_60_end_rho_plot = mask_60_end_rho_plot2*grid.mask_rho.values


# Seabed section 1
# Make an array of zeros
seabed_sec1_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 2
seabed_section1_idx = np.where(mask_0_10m_rho_plot == 2)
# Set these areas to 1 in the mask 
seabed_sec1_mask[seabed_section1_idx] = 1

# Seabed section 2
# Make an array of zeros
seabed_sec2_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 3
seabed_section2_idx = np.where(mask_0_10m_rho_plot == 3)
# Set these areas to 1 in the other plot
seabed_sec2_mask[seabed_section2_idx] = 1

# Seabed section 3
# Make an array of zeros
seabed_sec3_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 4
seabed_section3_idx = np.where(mask_0_10m_rho_plot == 4)
# Set these areas to 1 in the other plot
seabed_sec3_mask[seabed_section3_idx] = 1

# Seabed section 4
# Make an array of zeros
seabed_sec4_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 5
seabed_section4_idx = np.where(mask_0_10m_rho_plot == 5)
# Set these areas to 1 in the other plot
seabed_sec4_mask[seabed_section4_idx] = 1

# Seabed section 5
# Make an array of zeros
seabed_sec5_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 6
seabed_section5_idx = np.where(mask_10_30m_rho_plot == 6)
# Set these areas to 1 in the other plot
seabed_sec5_mask[seabed_section5_idx] = 1

# Seabed section 6
# Make an array of zeros
seabed_sec6_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 7
seabed_section6_idx = np.where(mask_10_30m_rho_plot == 7)
# Set these areas to 1 in the other plot
seabed_sec6_mask[seabed_section6_idx] = 1

# Seabed section 7
# Make an array of zeros
seabed_sec7_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 8
seabed_section7_idx = np.where(mask_10_30m_rho_plot == 8)
# Set these areas to 1 in the other plot
seabed_sec7_mask[seabed_section7_idx] = 1

# Seabed section 8
# Make an array of zeros
seabed_sec8_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 9
seabed_section8_idx = np.where(mask_10_30m_rho_plot == 9)
# Set these areas to 1 in the other plot
seabed_sec8_mask[seabed_section8_idx] = 1

# Seabed section 9
# Make an array of zeros
seabed_sec9_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 10
seabed_section9_idx = np.where(mask_30_60m_rho_plot == 10)
# Set these areas to 1 in the other plot
seabed_sec9_mask[seabed_section9_idx] = 1

# Seabed section 10
# Make an array of zeros
seabed_sec10_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 11
seabed_section10_idx = np.where(mask_30_60m_rho_plot == 11)
# Set these areas to 1 in the other plot
seabed_sec10_mask[seabed_section10_idx] = 1

# Seabed section 11
# Make an array of zeros
seabed_sec11_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 12
seabed_section11_idx = np.where(mask_30_60m_rho_plot == 12)
# Set these areas to 1 in the other plot
seabed_sec11_mask[seabed_section11_idx] = 1

# Seabed section 12
# Make an array of zeros
seabed_sec12_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 13
seabed_section12_idx = np.where(mask_30_60m_rho_plot == 13)
# Set these areas to 1 in the other plot
seabed_sec12_mask[seabed_section12_idx] = 1

# Seabed section 13
# Make an array of zeros
seabed_sec13_mask = np.zeros_like(grid.mask_rho)
# Get the indices where it equals 14
seabed_section13_idx1 = np.where(mask_60_end_rho_plot == 14)
seabed_section13_idx2 = np.where(mask_30_60m_rho_plot == 14)
seabed_section13_idx3 = np.where(mask_10_30m_rho_plot == 14)
seabed_section13_idx4 = np.where(mask_0_10m_rho_plot == 14)
# Set these areas to 1 in the other plot
seabed_sec13_mask[seabed_section13_idx1] = 1
seabed_sec13_mask[seabed_section13_idx2] = 1
seabed_sec13_mask[seabed_section13_idx3] = 1
seabed_sec13_mask[seabed_section13_idx4] = 1


# Okay now we have all of the masks for the different regions
# Need to multiply each sediment class (rivers and seabed) by masks of all regions
# for all time steps (loop through time) and sum (or average) over those regions 
# This will lead to 13 numbers per sediment class (26) that can then be used in 
# pie charts or bar graphs over time or something but it will be a lot of data
# so need to think about how to do this and how to organize it
# Is manual the best option? Maybe make a function that multiplies by all masks 
# and sums and returns the number? Then can call is once per sediment class? It
# will only return 13 numbers (for each time step) so this shouldn't be too crazy...

# Make a function that will take a given output of sediment class and the 
# masks for each section, and will multiply the kg sed by each mask and save
# to giant array for that mask?
def multiply_kg_sed_by_masks(depth_int_ssc, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask):
    """
    The purpose of this function it to take a given timeseries of kg of sediment class in 
    each grid cell of seabed over all time that is already summed over depth [time, eta, rho], 
    then multiply it by masks for the different seabed sections, then takes the sum over that 
    section, then saves this to a giant array and return that array that has a value for all 
    seabed sections for all time steps [time, seabed_section].
    

    Parameters
    ----------
    depth_int_ssc : Variable containing kssc for given sediment class [time, eta, xi]
    full_time_len : Number of time steps in the array/model output
    temp_mask : Variable containing the land mask of the grid 
    seabed_sec1_mask : Mask for Seabed Section 1
    seabed_sec2_mask : Mask for Seabed Section 2
    seabed_sec3_mask : Mask for Seabed Section 3
    seabed_sec4_mask : Mask for Seabed Section 4
    seabed_sec5_mask : Mask for Seabed Section 5
    seabed_sec6_mask : Mask for Seabed Section 6
    seabed_sec7_mask : Mask for Seabed Section 7
    seabed_sec8_mask : Mask for Seabed Section 8
    seabed_sec9_mask : Mask for Seabed Section 9
    seabed_sec10_mask : Mask for Seabed Section 10
    seabed_sec11_mask : Mask for Seabed Section 11
    seabed_sec12_mask : Mask for Seabed Section 12
    seabed_sec13_mask : Mask for Seabed Section 13

    Returns
    -------
    ssc_summed_masked : An array with a time series of total ssc of sediment class
    in each section [time, seabed_section]; Sec1 is index 0, sec2 is index 1, etc...

    """
    
    # Set the number of masks/sections
    num_seabed_sections = 13
    
    # Make an empty array to hold the output 
    ssc_summed_masked = np.empty((full_time_len, num_seabed_sections))
    
    # Loop through time, multiply by each mask, and save to array 
    for tt in range(full_time_len):
        # Section 1
        ssc_summed_masked[tt,0] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec1_mask, axis=(0,1))
        # Section 2
        ssc_summed_masked[tt,1] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec2_mask, axis=(0,1))
        # Section 3
        ssc_summed_masked[tt,2] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec3_mask, axis=(0,1))
        # Section 4
        ssc_summed_masked[tt,3] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec4_mask, axis=(0,1))
        # Section 5
        ssc_summed_masked[tt,4] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec5_mask, axis=(0,1))
        # Section 6
        ssc_summed_masked[tt,5] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec6_mask, axis=(0,1))
        # Section 7
        ssc_summed_masked[tt,6] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec7_mask, axis=(0,1))
        # Section 8
        ssc_summed_masked[tt,7] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec8_mask, axis=(0,1))
        # Section 9
        ssc_summed_masked[tt,8] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec9_mask, axis=(0,1))
        # Section 10
        ssc_summed_masked[tt,9] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec10_mask, axis=(0,1))
        # Section 11
        ssc_summed_masked[tt,10] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec11_mask, axis=(0,1))
        # Section 12
        ssc_summed_masked[tt,11] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec12_mask, axis=(0,1))
        # Section 13
        ssc_summed_masked[tt,12] = np.nansum(depth_int_ssc[tt,:,:]*temp_mask*seabed_sec13_mask, axis=(0,1))
        
        
    # Now that the array is filled with one value for each section for each time, 
    # return this array
    return(ssc_summed_masked)



# Now call the function for each sediment class
print('started masking', flush=True)
# Kalikpik
depth_int_ssc_summed_masked_kal = multiply_kg_sed_by_masks(depth_int_ssc_kal, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Colville
depth_int_ssc_summed_masked_col = multiply_kg_sed_by_masks(depth_int_ssc_col, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Sagavanirktok
depth_int_ssc_summed_masked_sag = multiply_kg_sed_by_masks(depth_int_ssc_sag, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Fish Creek
depth_int_ssc_summed_masked_fis = multiply_kg_sed_by_masks(depth_int_ssc_fis, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# # Sakonowyak
# depth_int_ssc_summed_masked_sak = multiply_kg_sed_by_masks(depth_int_ssc_sak, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
#                              seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
#                              seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
#                              seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Kuparuk
depth_int_ssc_summed_masked_kup = multiply_kg_sed_by_masks(depth_int_ssc_kup, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# # Putuligayuk
# depth_int_ssc_summed_masked_put = multiply_kg_sed_by_masks(depth_int_ssc_put, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
#                              seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
#                              seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
#                              seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Staines
depth_int_ssc_summed_masked_sta = multiply_kg_sed_by_masks(depth_int_ssc_sta, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Canning 
depth_int_ssc_summed_masked_can = multiply_kg_sed_by_masks(depth_int_ssc_can, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Katakturuk
depth_int_ssc_summed_masked_kat = multiply_kg_sed_by_masks(depth_int_ssc_kat, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Hulahula
depth_int_ssc_summed_masked_hul = multiply_kg_sed_by_masks(depth_int_ssc_hul, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Jago
depth_int_ssc_summed_masked_jag = multiply_kg_sed_by_masks(depth_int_ssc_jag, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# # Siksik
# depth_int_ssc_summed_masked_sik = multiply_kg_sed_by_masks(depth_int_ssc_sik, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
#                              seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
#                              seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
#                              seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 1
depth_int_ssc_summed_masked_sec1 = multiply_kg_sed_by_masks(depth_int_ssc_sec1, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 2
depth_int_ssc_summed_masked_sec2 = multiply_kg_sed_by_masks(depth_int_ssc_sec2, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 3
depth_int_ssc_summed_masked_sec3 = multiply_kg_sed_by_masks(depth_int_ssc_sec3, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 4
depth_int_ssc_summed_masked_sec4 = multiply_kg_sed_by_masks(depth_int_ssc_sec4, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 5
depth_int_ssc_summed_masked_sec5 = multiply_kg_sed_by_masks(depth_int_ssc_sec5, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 6
depth_int_ssc_summed_masked_sec6 = multiply_kg_sed_by_masks(depth_int_ssc_sec6, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 7
depth_int_ssc_summed_masked_sec7 = multiply_kg_sed_by_masks(depth_int_ssc_sec7, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 8 
depth_int_ssc_summed_masked_sec8 = multiply_kg_sed_by_masks(depth_int_ssc_sec8, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 9
depth_int_ssc_summed_masked_sec9 = multiply_kg_sed_by_masks(depth_int_ssc_sec9, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 10 
depth_int_ssc_summed_masked_sec10 = multiply_kg_sed_by_masks(depth_int_ssc_sec10, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 11
depth_int_ssc_summed_masked_sec11 = multiply_kg_sed_by_masks(depth_int_ssc_sec11, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 12
depth_int_ssc_summed_masked_sec12 = multiply_kg_sed_by_masks(depth_int_ssc_sec12, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)
# Section 13
depth_int_ssc_summed_masked_sec13 = multiply_kg_sed_by_masks(depth_int_ssc_sec13, full_time_len, temp_mask, seabed_sec1_mask, seabed_sec2_mask,
                             seabed_sec3_mask, seabed_sec4_mask, seabed_sec5_mask, seabed_sec6_mask,
                             seabed_sec7_mask, seabed_sec8_mask, seabed_sec9_mask, seabed_sec10_mask,
                             seabed_sec11_mask, seabed_sec12_mask, seabed_sec13_mask)

print('done masking', flush=True)


# Now that we have the summed depth-integrated SSC for each class in each seabed 
# section, group them into seabed sections so that stacked bar charts of each 
# section can be made over time
# sec1_depth_int_ssc_sum_comp = [depth_int_ssc_summed_masked_kal[:,0], depth_int_ssc_summed_masked_col[:,0], depth_int_ssc_summed_masked_sag[:,0],
#                                depth_int_ssc_summed_masked_fis[:,0], depth_int_ssc_summed_masked_sak[:,0], depth_int_ssc_summed_masked_kup[:,0],
#                                depth_int_ssc_summed_masked_put[:,0], depth_int_ssc_summed_masked_sta[:,0], depth_int_ssc_summed_masked_can[:,0],
#                                depth_int_ssc_summed_masked_kat[:,0], depth_int_ssc_summed_masked_hul[:,0], depth_int_ssc_summed_masked_jag[:,0],
#                                depth_int_ssc_summed_masked_sik[:,0], depth_int_ssc_summed_masked_sec1[:,0], depth_int_ssc_summed_masked_sec2[:,0],
#                                depth_int_ssc_summed_masked_sec3[:,0], depth_int_ssc_summed_masked_sec4[:,0], depth_int_ssc_summed_masked_sec5[:,0],
#                                depth_int_ssc_summed_masked_sec6[:,0], depth_int_ssc_summed_masked_sec7[:,0], depth_int_ssc_summed_masked_sec8[:,0],
#                                depth_int_ssc_summed_masked_sec9[:,0], depth_int_ssc_summed_masked_sec10[:,0], depth_int_ssc_summed_masked_sec11[:,0],
#                                depth_int_ssc_summed_masked_sec12[:,0], depth_int_ssc_summed_masked_sec13[:,0]]
sec1_depth_int_ssc_sum_comp = [depth_int_ssc_summed_masked_kal[:,0], depth_int_ssc_summed_masked_col[:,0], depth_int_ssc_summed_masked_sag[:,0],
                               depth_int_ssc_summed_masked_fis[:,0], depth_int_ssc_summed_masked_kup[:,0],
                               depth_int_ssc_summed_masked_sta[:,0], depth_int_ssc_summed_masked_can[:,0],
                               depth_int_ssc_summed_masked_kat[:,0], depth_int_ssc_summed_masked_hul[:,0], depth_int_ssc_summed_masked_jag[:,0],
                               depth_int_ssc_summed_masked_sec1[:,0], depth_int_ssc_summed_masked_sec2[:,0],
                               depth_int_ssc_summed_masked_sec3[:,0], depth_int_ssc_summed_masked_sec4[:,0], depth_int_ssc_summed_masked_sec5[:,0],
                               depth_int_ssc_summed_masked_sec6[:,0], depth_int_ssc_summed_masked_sec7[:,0], depth_int_ssc_summed_masked_sec8[:,0],
                               depth_int_ssc_summed_masked_sec9[:,0], depth_int_ssc_summed_masked_sec10[:,0], depth_int_ssc_summed_masked_sec11[:,0],
                               depth_int_ssc_summed_masked_sec12[:,0], depth_int_ssc_summed_masked_sec13[:,0]]




# Make a list of labels in order
labels_tmp = ['Kalikpik', 'Fish Creek', 'Colville', 'Sakonowyak', 
          'Kuparuk', 'Putuligayuk', 'Sagavanirktok', 'Staines', 'Canning', 'Katakturuk', 'Hulahula',
          'Jago', 'Siksik', 'Section 1', 'Section 2', 'Section 3', 'Section 4',
          'Section 5', 'Section 8', 'Section 9', 'Section 10',
          'Section 11', 'Section 12']

labels_tmp2 = ['Kalikpik', 'Colville', 'Sagavanirktok', 'Fish Creek', 'Sakonowyak', 
          'Kuparuk', 'Putuligayuk', 'Staines', 'Canning', 'Katakturuk', 'Hulahula',
          'Jago', 'Siksik', 'Section 1', 'Section 2', 'Section 3', 'Section 4',
          'Section 5', 'Section 8', 'Section 9', 'Section 10',
          'Section 11', 'Section 12']

labels_all_13rivs = ['Kalikpik', 'Fish Creek', 'Colville', 'Sakonowyak', 
          'Kuparik', 'Putuligayuk', 'Sagavanirktok', 'Staines', 'Canning', 'Katakturuk', 'Hulahula',
          'Jago', 'Siksik', 'Section 1', 'Section 2', 'Section 3', 'Section 4',
          'Section 5', 'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
          'Section 11', 'Section 12', 'Section 13']

# Same as above but with Putuligayuk, Sakonowyak, and Siksik removed
labels_all = ['Kalikpik', 'Fish Creek', 'Colville', 
          'Kuparik', 'Sagavanirktok', 'Staines', 'Canning', 'Katakturuk', 'Hulahula',
          'Jago', 'Section 1', 'Section 2', 'Section 3', 'Section 4',
          'Section 5', 'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
          'Section 11', 'Section 12', 'Section 13']


# Make a list of all colors in order (rivers then seabed sections)
#colors_tmp = ['r', 'brown', 'deepskyblue', 'orange', 'green', 'b', 'dodgerblue', 'gold',
 #               'darkorange', 'aquamarine', 'blueviolet', 'magenta', 'deeppink', 'cornflowerblue', 
  #              'lightsteelblue', 'sandybrown', 'forestgreen', 'orangered',
   #             'hotpink', 'pink', 'lightgray', 'yellowgreen',
     #                            'mediumturquoise']

# =============================================================================
# colors_all = ['r', 'brown', 'deepskyblue', 'orange', 'green', 'b', 'dodgerblue', 'gold',
#                 'darkorange', 'aquamarine', 'blueviolet', 'magenta', 'deeppink', 'cornflowerblue', 
#                 'lightsteelblue', 'sandybrown', 'forestgreen', 'orangered',
#                 'lightsalmon', 'sienna', 'hotpink', 'pink', 'lightgray', 'yellowgreen',
#                                  'mediumturquoise', 'powderblue' ]
# =============================================================================

# Make a list of seabed colors
seabed_colors = ['cornflowerblue', 'lightsteelblue', 'sandybrown', 'forestgreen', 'orangered',
                 'lightsalmon', 'sienna', 'hotpink', 'pink', 'lightgray', 'yellowgreen',
                 'mediumturquoise', 'powderblue']

# Make a list of colors for the rivers 
river_colors = ['r', 'brown', 'deepskyblue', 'orange', 'green', 'b', 'dodgerblue', 'gold',
                'darkorange', 'aquamarine', 'blueviolet', 'magenta', 'deeppink']

# Make list of colors to use for plots
seabed_section_colors = ['#D1C8FB', '#F4B6D3', '#FFC8B0', '#FFE3B2', '#A190F4', 
                         '#E86DA6', '#FF925F', '#FFC863', '#785EF0', '#DC267F', 
                         '#FF6100', '#FFB000','#6490FF']
#river_marker_colors = ['#FC440F', '#00A6A6', '#5EF38C', '#26532B', '#0115F5',
               #        '#9C00A8', '#F43ECF', '#F5ED00']
river_marker_colors = ['#FC440F', '#F5ED00', '#5EF38C', '#26532B', '#F43ECF', '#9C00A8',
                       '#0115F5', '#00A6A6', '#AB64EB', '#D44179', '#08E0E3', '#B27009', '#EA8D40']

# New colors tmp
# River Order: Kalikpik, Fish Creek, Colville, Sakonowyak, Kuparuk, Putuligayuk, 
# Sagavanirktok, Staines, Canning, Katakturuk, Hulahula, Jago, Siksik, 
# Section 1 ..., Section 13 (sections are in order)
colors_tmp_13rivs = ['#FC440F', '#F5ED00', '#5EF38C', '#26532B', '#F43ECF', '#9C00A8',
              '#0115F5', '#00A6A6', '#AB64EB', '#D44179', '#08E0E3', '#B27009', '#EA8D40',
              '#D1C8FB', '#F4B6D3', '#FFC8B0', '#FFE3B2', '#A190F4', 
              '#E86DA6', '#FF925F', '#FFC863', '#785EF0', '#DC267F',
              '#FF6100', '#FFB000','#6490FF'] 

# Same as above but with Putuligayuk, Sakonowyak, and Siksik removed 
colors_tmp = ['#FC440F', '#F5ED00', '#5EF38C', '#F43ECF',
              '#0115F5', '#00A6A6', '#AB64EB', '#D44179', '#08E0E3', '#B27009',
              '#D1C8FB', '#F4B6D3', '#FFC8B0', '#FFE3B2', '#A190F4', 
              '#E86DA6', '#FF925F', '#FFC863', '#785EF0', '#DC267F',
              '#FF6100', '#FFB000','#6490FF'] 


# New colors tmp2 - with colors for sections 6, 7, 13 removed since we are ignoring those for now
colors_tmp2 = ['#FC440F', '#F5ED00', '#5EF38C', '#26532B', '#F43ECF', '#9C00A8',
              '#0115F5', '#00A6A6', '#AB64EB', '#D44179', '#08E0E3', '#B27009', '#EA8D40',
              '#D1C8FB', '#F4B6D3', '#FFC8B0', '#FFE3B2', '#A190F4', 
              '#FFC863', '#785EF0', '#DC267F',
              '#FF6100', '#FFB000']

# Kalikpik
# Fish Creek
# Colville
# Sakonowyak
# Kuparuk
# Putuligayuk
# Sagavanirktok
# Staines
# Canning 
# Katakturuk
# Hulahula
# Jago
# Siksik

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
print('Time step (736:737): ', time_steps[736:737], flush=True)
print('Time step (738): ', time_steps[738], flush=True)
print('Time step (740): ', time_steps[740], flush=True)


# ---------------- Trim to Time we Trust --------------------
# Trim all of these variables to the time that we trust 
# Kalikpik
depth_int_ssc_summed_masked_kal = depth_int_ssc_summed_masked_kal[:738,:]
# Colville
depth_int_ssc_summed_masked_col = depth_int_ssc_summed_masked_col[:738,:]
# Sagavanirktok
depth_int_ssc_summed_masked_sag = depth_int_ssc_summed_masked_sag[:738,:]
# Fish Creek
depth_int_ssc_summed_masked_fis = depth_int_ssc_summed_masked_fis[:738,:]
# # Sakonowyak
# depth_int_ssc_summed_masked_sak = depth_int_ssc_summed_masked_sak[:738,:]
# Kuparuk
depth_int_ssc_summed_masked_kup = depth_int_ssc_summed_masked_kup[:738,:]
# # Putuligayuk
# depth_int_ssc_summed_masked_put = depth_int_ssc_summed_masked_put[:738,:]
# Staines
depth_int_ssc_summed_masked_sta = depth_int_ssc_summed_masked_sta[:738,:]
# Canning
depth_int_ssc_summed_masked_can = depth_int_ssc_summed_masked_can[:738,:]
# Katakturuk
depth_int_ssc_summed_masked_kat = depth_int_ssc_summed_masked_kat[:738,:]
# Hulahula
depth_int_ssc_summed_masked_hul = depth_int_ssc_summed_masked_hul[:738,:]
# Jago
depth_int_ssc_summed_masked_jag = depth_int_ssc_summed_masked_jag[:738,:]
# # Siksik
# depth_int_ssc_summed_masked_sik = depth_int_ssc_summed_masked_sik[:738,:]
# Section 1 
depth_int_ssc_summed_masked_sec1 = depth_int_ssc_summed_masked_sec1[:738,:]
# Section 2 
depth_int_ssc_summed_masked_sec2 = depth_int_ssc_summed_masked_sec2[:738,:]
# Section 3 
depth_int_ssc_summed_masked_sec3 = depth_int_ssc_summed_masked_sec3[:738,:]
# Section 4 
depth_int_ssc_summed_masked_sec4 = depth_int_ssc_summed_masked_sec4[:738,:]
# Section 5 
depth_int_ssc_summed_masked_sec5 = depth_int_ssc_summed_masked_sec5[:738,:]
# Section 6 
depth_int_ssc_summed_masked_sec6 = depth_int_ssc_summed_masked_sec6[:738,:]
# Section 7 
depth_int_ssc_summed_masked_sec7 = depth_int_ssc_summed_masked_sec7[:738,:]
# Section 8 
depth_int_ssc_summed_masked_sec8 = depth_int_ssc_summed_masked_sec8[:738,:]
# Section 9 
depth_int_ssc_summed_masked_sec9 = depth_int_ssc_summed_masked_sec9[:738,:]
# Section 10 
depth_int_ssc_summed_masked_sec10 = depth_int_ssc_summed_masked_sec10[:738,:]
# Section 11 
depth_int_ssc_summed_masked_sec11 = depth_int_ssc_summed_masked_sec11[:738,:]
# Section 12 
depth_int_ssc_summed_masked_sec12 = depth_int_ssc_summed_masked_sec12[:738,:]
# Section 13 
depth_int_ssc_summed_masked_sec13 = depth_int_ssc_summed_masked_sec13[:738,:]

# Trim time steps to match
time_steps = time_steps[:738]


# ---- Percent Over Time ----
# Prep the data to be the percentage of local resuspension over time

# Make a function to calculate the percentage over time 
def get_percent_local_over_time(depth_int_ssc_summed_masked_of_interest, idx, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13):
    """
    This function takes a given array of depth-integrated suspended sediment concentration 
    for a certain region of interest and divides it by the total value in that region
    for all time to give a time series of the percentage of locally resuspended suspended.

    Inputs:
    - depth_int_ssc_summed_masked_of_interest: Time series/Array of the amount of depth-integrated 
       suspended sediment from a certain source in a certain region over time 
    - idx: Index position in giant arrays that corresponds to the section we want to look at
    - depth_int_ssc_summed_masked_*: Time series/Array of the amount of depth-integrated 
       suspended sediment from a certain source in a certain region over time 
    
    Outputs:
    - percent_local_resuspended: Time series of the percent of depth-integrated suspended 
       sediment that is locally resuspended from that region 
    """

    # Sum up all of the suspended sediment sources in this space 
    tot_ssc_in_region_of_interest = depth_int_ssc_summed_masked_kal[:,idx]+depth_int_ssc_summed_masked_fis[:,idx]+depth_int_ssc_summed_masked_col[:,idx]+depth_int_ssc_summed_masked_kup[:,idx]+depth_int_ssc_summed_masked_sag[:,idx]+depth_int_ssc_summed_masked_sta[:,idx]+depth_int_ssc_summed_masked_can[:,idx]+depth_int_ssc_summed_masked_kat[:,idx]+depth_int_ssc_summed_masked_hul[:,idx]+depth_int_ssc_summed_masked_jag[:,idx]+depth_int_ssc_summed_masked_sec1[:,idx]+depth_int_ssc_summed_masked_sec2[:,idx]+depth_int_ssc_summed_masked_sec3[:,idx]+depth_int_ssc_summed_masked_sec4[:,idx]+depth_int_ssc_summed_masked_sec5[:,idx]+depth_int_ssc_summed_masked_sec6[:,idx]+depth_int_ssc_summed_masked_sec7[:,idx]+depth_int_ssc_summed_masked_sec8[:,idx]+depth_int_ssc_summed_masked_sec9[:,idx]+depth_int_ssc_summed_masked_sec10[:,idx]+depth_int_ssc_summed_masked_sec11[:,idx]+depth_int_ssc_summed_masked_sec12[:,idx]+depth_int_ssc_summed_masked_sec13[:,idx]

    # Calculate the percent that is locally resuspended
    percent_local_resuspended = (depth_int_ssc_summed_masked_of_interest[:,idx]/tot_ssc_in_region_of_interest)*100
    # Check this 
    print('depth_int_ssc_summed_masked_of_interest[24,idx]: ', depth_int_ssc_summed_masked_of_interest[24,idx])
    print('tot_ssc_in_region_of_interest[24]: ', tot_ssc_in_region_of_interest[24])
    print('percent_local_resuspended[24]: ', percent_local_resuspended[24])

    # Return this value...
    return(percent_local_resuspended)

# Make a version that is instead just the percent suspended 
# sediment that is locally resuspended from Section 1 with one
# value for each time (so divide the amount from section 1 by the total)
# **Check units because I think we want kg and then take percent? so it is 
# percent mass?** --> so the units are kg/m2 so we could multiply by dxdy
# but when we divide the two this will just cancel out anyway so there is
# no need to do that
# depth_int_ssc_summed_masked_sec1 = kg/m2 

# Call the function for Section 1
percent_local_resuspended_over_time_sec1 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec1, 0, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 2
percent_local_resuspended_over_time_sec2 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec2, 1, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 3
percent_local_resuspended_over_time_sec3 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec3, 2, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 4
percent_local_resuspended_over_time_sec4 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec4, 3, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 5
percent_local_resuspended_over_time_sec5 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec5, 4, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 6
percent_local_resuspended_over_time_sec6 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec6, 5, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 7
percent_local_resuspended_over_time_sec7 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec7, 6, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 8
percent_local_resuspended_over_time_sec8 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec8, 7, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 9
percent_local_resuspended_over_time_sec9 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec9, 8, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 10
percent_local_resuspended_over_time_sec10 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec10, 9, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 11
percent_local_resuspended_over_time_sec11 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec11, 10, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 12
percent_local_resuspended_over_time_sec12 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec12, 11, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)

# Call the function for Section 13
percent_local_resuspended_over_time_sec13 = get_percent_local_over_time(depth_int_ssc_summed_masked_sec13, 12, depth_int_ssc_summed_masked_kal, depth_int_ssc_summed_masked_fis, depth_int_ssc_summed_masked_col, depth_int_ssc_summed_masked_kup, depth_int_ssc_summed_masked_sag, depth_int_ssc_summed_masked_sta, depth_int_ssc_summed_masked_can, depth_int_ssc_summed_masked_kat, depth_int_ssc_summed_masked_hul, depth_int_ssc_summed_masked_jag, depth_int_ssc_summed_masked_sec1, depth_int_ssc_summed_masked_sec2, depth_int_ssc_summed_masked_sec3, depth_int_ssc_summed_masked_sec4, depth_int_ssc_summed_masked_sec5, depth_int_ssc_summed_masked_sec6, depth_int_ssc_summed_masked_sec7, depth_int_ssc_summed_masked_sec8, depth_int_ssc_summed_masked_sec9, depth_int_ssc_summed_masked_sec10, depth_int_ssc_summed_masked_sec11, depth_int_ssc_summed_masked_sec12, depth_int_ssc_summed_masked_sec13)


# Save these to a netcdf since the analysis takes so long?
percent_local_resuspended_over_time = xr.Dataset(
    data_vars=dict(
        percent_local_resuspended_over_time_sec1=(['ocean_time'], percent_local_resuspended_over_time_sec1),
        percent_local_resuspended_over_time_sec2=(['ocean_time'], percent_local_resuspended_over_time_sec2),
        percent_local_resuspended_over_time_sec3=(['ocean_time'], percent_local_resuspended_over_time_sec3),
        percent_local_resuspended_over_time_sec4=(['ocean_time'], percent_local_resuspended_over_time_sec4),
        percent_local_resuspended_over_time_sec5=(['ocean_time'], percent_local_resuspended_over_time_sec5),
        percent_local_resuspended_over_time_sec6=(['ocean_time'], percent_local_resuspended_over_time_sec6),
        percent_local_resuspended_over_time_sec7=(['ocean_time'], percent_local_resuspended_over_time_sec7),
        percent_local_resuspended_over_time_sec8=(['ocean_time'], percent_local_resuspended_over_time_sec8),
        percent_local_resuspended_over_time_sec9=(['ocean_time'], percent_local_resuspended_over_time_sec9),
        percent_local_resuspended_over_time_sec10=(['ocean_time'], percent_local_resuspended_over_time_sec10),
        percent_local_resuspended_over_time_sec11=(['ocean_time'], percent_local_resuspended_over_time_sec11),
        percent_local_resuspended_over_time_sec12=(['ocean_time'], percent_local_resuspended_over_time_sec12),
        percent_local_resuspended_over_time_sec13=(['ocean_time'], percent_local_resuspended_over_time_sec13),
    ),
    coords=dict(
        ocean_time=('ocean_time', time_steps)
    ),
    attrs=dict(description='Time-series ROMS output of percent of locally resuspended sediment in each section')
)

# Save to a netcdf
# Aggregated 
percent_local_resuspended_over_time.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_local_resuspension/percent_local_resuspended_over_time_aggregated.nc')
# Unaggregated 
#percent_local_resuspended_over_time.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_local_resuspension/percent_local_resuspended_over_time_unaggregated.nc')


# Make list of colors to use for plots
seabed_section_colors = ['#D1C8FB', '#F4B6D3', '#FFC8B0', '#FFE3B2', '#A190F4', 
                         '#E86DA6', '#FF925F', '#FFC863', '#785EF0', '#DC267F', 
                         '#FF6100', '#FFB000','#6490FF']



# # --------------------------------------------------------------------------------
# # --------------- Plot 1: Box and Whisker Plot for Section(s)... -----------------
# # --------------------------------------------------------------------------------
# Make a box and whiskers plot of the percent of local resuspension over time
# in multiple sections, one plot for each section 

# Make the figure 
fig1, ax1 = plt.subplots(figsize=(16,8))

# Make labels
section_names_long = ['Section 1', 'Section 2', 'Section 3', 'Section 4', 'Section 5', 
                 'Section 6', 'Section 7', 'Section 8', 'Section 9', 'Section 10',
                 'Section 11', 'Section 12', 'Section 13']
section_names_short = ['1', '2', '3', '4', '5', 
                 '6', '7', '8', '9', '10',
                 '11', '12', '13']

# Combine the data into a list
local_resusp_all_sections = [percent_local_resuspended_over_time_sec1, percent_local_resuspended_over_time_sec2,
                             percent_local_resuspended_over_time_sec3, percent_local_resuspended_over_time_sec4,
                             percent_local_resuspended_over_time_sec5, percent_local_resuspended_over_time_sec6,
                             percent_local_resuspended_over_time_sec7, percent_local_resuspended_over_time_sec8,
                             percent_local_resuspended_over_time_sec9, percent_local_resuspended_over_time_sec10,
                             percent_local_resuspended_over_time_sec11, percent_local_resuspended_over_time_sec12,
                             percent_local_resuspended_over_time_sec13]

# Plot?
ax1.boxplot(local_resusp_all_sections[0], labels=section_names_short[0])

# Label the plot
ax1.set_xlabel('Section', fontsize=fontsize-2)
ax1.set_ylabel('Percent Locally Resuspended', fontsize=fontsize-2)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_local_resuspension/percent_local_resuspension_all_sections_test_001.png', transparent=True, bbox_inches='tight')


# # Sample data for multiple boxplots
# data1 = np.random.normal(0, 1, 100)
# data2 = np.random.normal(2, 1.5, 100)
# data3 = np.random.normal(-1, 0.8, 100)

# # Combine data into a list
# data_to_plot = [data1, data2, data3]

# # Create the boxplot
# plt.figure(figsize=(8, 6))
# plt.boxplot(data_to_plot, labels=['Group 1', 'Group 2', 'Group 3'])
# plt.title('Multiple Boxplots on Same Axes')
# plt.xlabel('Group')
# plt.ylabel('Value')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()



# # --------------------------------------------------------------------------------
# # --------------- Plot 2: Line Plot for Section(s)... -----------------
# # --------------------------------------------------------------------------------
# Try to make a line plot for the sections 

# Make the figure 
fig2, ax2 = plt.subplots(figsize=(16,8))


# Plot each section as a line 
ax2.plot(time_steps, percent_local_resuspended_over_time_sec1, color=seabed_section_colors[0], label=section_names_long[0], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec2, color=seabed_section_colors[1], label=section_names_long[1], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec3, color=seabed_section_colors[2], label=section_names_long[2], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec4, color=seabed_section_colors[3], label=section_names_long[3], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec5, color=seabed_section_colors[4], label=section_names_long[4], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec6, color=seabed_section_colors[5], label=section_names_long[5], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec7, color=seabed_section_colors[6], label=section_names_long[6], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec8, color=seabed_section_colors[7], label=section_names_long[7], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec9, color=seabed_section_colors[8], label=section_names_long[8], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec10, color=seabed_section_colors[9], label=section_names_long[9], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec11, color=seabed_section_colors[10], label=section_names_long[10], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec12, color=seabed_section_colors[11], label=section_names_long[11], linewidth=3)
ax2.plot(time_steps, percent_local_resuspended_over_time_sec13, color=seabed_section_colors[12], label=section_names_long[12], linewidth=3)



# Label the plot
ax2.set_xlabel('Time', fontsize=fontsize-2)
ax2.set_ylabel('Percent Locally Resuspended', fontsize=fontsize-2)
ax2.legend(fontsize=fontsize-3, ncols=3)
#plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
#plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_local_resuspension/percent_local_resuspension_timeseries_test_001.png', transparent=True, bbox_inches='tight')









# Check this before continuing...
# **PICK UP HERE TOMORROW WITH TESTING AND CHANGING AND REORGANIZING**





# # --------------------------------------------------------------------------------
# # --------------------- Plot 1: Stacked Bar Charts for Section 1 -----------------
# # ----------------------------------- Timeseries? --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec1_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,0],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,0],
#              'Colville': depth_int_ssc_summed_masked_col[:,0],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,0],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,0],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,0],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,0],
#              'Staines': depth_int_ssc_summed_masked_sta[:,0],
#              'Canning': depth_int_ssc_summed_masked_can[:,0],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,0],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,0],
#              'Jago': depth_int_ssc_summed_masked_jag[:,0],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,0],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,0],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,0],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,0],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,0],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,0],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,0],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,0],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,0],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,0],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,0],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,0],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,0], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,0]}


# # Make a dataframe 
# sec1_df = pd.DataFrame(sec1_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig1, ax1 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax1.bar(sec1_df['Date'], sec1_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec1_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax1.set_ylim([0,4000])
# ax1.set_xlabel('Date', fontsize=fontsize)
# ax1.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax1.set_title('Total SSC in Section 1 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec1_2020_aggregated_dbsed0001_full_0003.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec1_2020_unaggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 2: Stacked Bar Charts for Section 2 -----------------
# # ----------------------------------- Timeseries? --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec2_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,1],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,1],
#              'Colville': depth_int_ssc_summed_masked_col[:,1],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,1],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,1],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,1],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,1],
#              'Staines': depth_int_ssc_summed_masked_sta[:,1],
#              'Canning': depth_int_ssc_summed_masked_can[:,1],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,1],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,1],
#              'Jago': depth_int_ssc_summed_masked_jag[:,1],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,1],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,1],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,1],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,1],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,1],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,1],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,1],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,1],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,1],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,1],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,1],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,1],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,1], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,1]}

# # Make a dataframe 
# sec2_df = pd.DataFrame(sec2_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig2, ax2 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax2.bar(sec2_df['Date'], sec2_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec2_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax2.set_xlabel('Date', fontsize=fontsize)
# ax2.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax2.set_title('Total SSC in Section 2 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec2_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec2_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 3: Stacked Bar Charts for Section 3 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec3_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,2],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,2],
#              'Colville': depth_int_ssc_summed_masked_col[:,2],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,2],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,2],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,2],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,2],
#              'Staines': depth_int_ssc_summed_masked_sta[:,2],
#              'Canning': depth_int_ssc_summed_masked_can[:,2],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,2],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,2],
#              'Jago': depth_int_ssc_summed_masked_jag[:,2],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,2],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,2],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,2],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,2],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,2],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,2],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,2],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,2],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,2],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,2],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,2],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,2],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,2], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,2]}

# # Make a dataframe 
# sec3_df = pd.DataFrame(sec3_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig3, ax3 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax3.bar(sec3_df['Date'], sec3_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec3_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax3.set_xlabel('Date', fontsize=fontsize)
# ax3.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax3.set_title('Total SSC in Section 3 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec3_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec3_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 4: Stacked Bar Charts for Section 4 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec4_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,3],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,3],
#              'Colville': depth_int_ssc_summed_masked_col[:,3],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,3],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,3],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,3],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,3],
#              'Staines': depth_int_ssc_summed_masked_sta[:,3],
#              'Canning': depth_int_ssc_summed_masked_can[:,3],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,3],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,3],
#              'Jago': depth_int_ssc_summed_masked_jag[:,3],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,3],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,3],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,3],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,3],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,3],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,3],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,3],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,3],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,3],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,3],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,3],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,3],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,3], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,3]}

# # Make a dataframe 
# sec4_df = pd.DataFrame(sec4_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig4, ax4 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax4.bar(sec4_df['Date'], sec4_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec4_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax4.set_xlabel('Date', fontsize=fontsize)
# ax4.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax4.set_title('Total SSC in Section 4 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec4_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec4_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 5: Stacked Bar Charts for Section 5 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec5_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,4],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,4],
#              'Colville': depth_int_ssc_summed_masked_col[:,4],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,4],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,4],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,4],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,4],
#              'Staines': depth_int_ssc_summed_masked_sta[:,4],
#              'Canning': depth_int_ssc_summed_masked_can[:,4],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,4],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,4],
#              'Jago': depth_int_ssc_summed_masked_jag[:,4],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,4],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,4],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,4],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,4],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,4],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,4],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,4],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,4],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,4],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,4],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,4],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,4],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,4], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,4]}

# # Make a dataframe 
# sec5_df = pd.DataFrame(sec5_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig5, ax5 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax5.bar(sec5_df['Date'], sec5_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec5_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax5.set_xlabel('Date', fontsize=fontsize)
# ax5.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax5.set_title('Total SSC in Section 5 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec5_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec5_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 6: Stacked Bar Charts for Section 6 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec6_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,5],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,5],
#              'Colville': depth_int_ssc_summed_masked_col[:,5],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,5],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,5],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,5],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,5],
#              'Staines': depth_int_ssc_summed_masked_sta[:,5],
#              'Canning': depth_int_ssc_summed_masked_can[:,5],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,5],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,5],
#              'Jago': depth_int_ssc_summed_masked_jag[:,5],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,5],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,5],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,5],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,5],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,5],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,5],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,5],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,5],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,5],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,5],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,5],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,5],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,5], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,5]}

# # Make a dataframe 
# sec6_df = pd.DataFrame(sec6_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig6, ax6 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax6.bar(sec6_df['Date'], sec6_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec6_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax6.set_xlabel('Date', fontsize=fontsize)
# ax6.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax6.set_title('Total SSC in Section 6 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec6_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec6_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 7: Stacked Bar Charts for Section 7 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec7_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,6],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,6],
#              'Colville': depth_int_ssc_summed_masked_col[:,6],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,6],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,6],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,6],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,6],
#              'Staines': depth_int_ssc_summed_masked_sta[:,6],
#              'Canning': depth_int_ssc_summed_masked_can[:,6],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,6],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,6],
#              'Jago': depth_int_ssc_summed_masked_jag[:,6],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,6],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,6],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,6],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,6],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,6],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,6],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,6],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,6],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,6],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,6],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,6],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,6],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,6], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,6]}

# # Make a dataframe 
# sec7_df = pd.DataFrame(sec7_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig7, ax7 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax7.bar(sec7_df['Date'], sec7_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec7_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax7.set_xlabel('Date', fontsize=fontsize)
# ax7.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax7.set_title('Total SSC in Section 7 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec7_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec7_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 8: Stacked Bar Charts for Section 8 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec8_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,7],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,7],
#              'Colville': depth_int_ssc_summed_masked_col[:,7],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,7],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,7],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,7],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,7],
#              'Staines': depth_int_ssc_summed_masked_sta[:,7],
#              'Canning': depth_int_ssc_summed_masked_can[:,7],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,7],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,7],
#              'Jago': depth_int_ssc_summed_masked_jag[:,7],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,7],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,7],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,7],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,7],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,7],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,7],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,7],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,7],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,7],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,7],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,7],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,7],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,7], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,7]}

# # Make a dataframe 
# sec8_df = pd.DataFrame(sec8_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig8, ax8 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax8.bar(sec8_df['Date'], sec8_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec8_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax8.set_xlabel('Date', fontsize=fontsize)
# ax8.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax8.set_title('Total SSC in Section 8 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec8_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec8_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 9: Stacked Bar Charts for Section 9 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec9_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,8],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,8],
#              'Colville': depth_int_ssc_summed_masked_col[:,8],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,8],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,8],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,8],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,8],
#              'Staines': depth_int_ssc_summed_masked_sta[:,8],
#              'Canning': depth_int_ssc_summed_masked_can[:,8],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,8],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,8],
#              'Jago': depth_int_ssc_summed_masked_jag[:,8],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,8],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,8],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,8],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,8],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,8],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,8],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,8],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,8],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,8],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,8],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,8],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,8],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,8], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,8]}

# # Make a dataframe 
# sec9_df = pd.DataFrame(sec9_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig9, ax9 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax9.bar(sec9_df['Date'], sec9_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec9_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax9.set_xlabel('Date', fontsize=fontsize)
# ax9.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax9.set_title('Total SSC in Section 9 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec9_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec9_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 10: Stacked Bar Charts for Section 10 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 1

# # Prep the data
# sec10_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,9],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,9],
#              'Colville': depth_int_ssc_summed_masked_col[:,9],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,9],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,9],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,9],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,9],
#              'Staines': depth_int_ssc_summed_masked_sta[:,9],
#              'Canning': depth_int_ssc_summed_masked_can[:,9],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,9],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,9],
#              'Jago': depth_int_ssc_summed_masked_jag[:,9],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,9],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,9],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,9],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,9],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,9],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,9],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,9],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,9],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,9],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,9],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,9],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,9],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,9], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,9]}

# # Make a dataframe 
# sec10_df = pd.DataFrame(sec10_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig10, ax10 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax10.bar(sec10_df['Date'], sec10_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec10_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax10.set_xlabel('Date', fontsize=fontsize)
# ax10.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax10.set_title('Total SSC in Section 10 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec10_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec10_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 11: Stacked Bar Charts for Section 11 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 11

# # Prep the data
# sec11_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,10],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,10],
#              'Colville': depth_int_ssc_summed_masked_col[:,10],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,10],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,10],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,10],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,10],
#              'Staines': depth_int_ssc_summed_masked_sta[:,10],
#              'Canning': depth_int_ssc_summed_masked_can[:,10],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,10],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,10],
#              'Jago': depth_int_ssc_summed_masked_jag[:,10],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,10],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,10],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,10],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,10],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,10],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,10],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,10],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,10],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,10],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,10],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,10],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,10],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,10], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,10]}

# # Make a dataframe 
# sec11_df = pd.DataFrame(sec11_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig11, ax11 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax11.bar(sec11_df['Date'], sec11_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec11_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax11.set_xlabel('Date', fontsize=fontsize)
# ax11.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax11.set_title('Total SSC in Section 11 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec11_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec11_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 12: Stacked Bar Charts for Section 12 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 11

# # Prep the data
# sec12_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,11],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,11],
#              'Colville': depth_int_ssc_summed_masked_col[:,11],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,11],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,11],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,11],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,11],
#              'Staines': depth_int_ssc_summed_masked_sta[:,11],
#              'Canning': depth_int_ssc_summed_masked_can[:,11],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,11],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,11],
#              'Jago': depth_int_ssc_summed_masked_jag[:,11],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,11],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,11],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,11],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,11],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,11],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,11],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,11],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,11],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,11],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,11],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,11],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,11],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,11], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,11]}

# # Make a dataframe 
# sec12_df = pd.DataFrame(sec12_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig12, ax12 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax12.bar(sec12_df['Date'], sec12_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec12_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)
# ax12.set_xlabel('Date', fontsize=fontsize)
# ax12.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax12.set_title('Total SSC in Section 12 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec12_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec12_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)

# # --------------------------------------------------------------------------------
# # --------------------- Plot 13: Stacked Bar Charts for Section 13 -----------------
# # ----------------------------------- Timeseries --------------------------------
# # --------------------------------------------------------------------------------
# # Make a time series of stacked bar charts where each color in stack is a 
# # different sediment class of summed depth-integrated SSC in Section 11

# # Prep the data
# sec13_data = {'Date': time_steps, # 'Date': time_tmp.astype(float)
#              'Kalikpik': depth_int_ssc_summed_masked_kal[:,12],
#              'Fish Creek': depth_int_ssc_summed_masked_fis[:,12],
#              'Colville': depth_int_ssc_summed_masked_col[:,12],
#              #'Sakonowyak': depth_int_ssc_summed_masked_sak[:,12],
#              'Kuparuk': depth_int_ssc_summed_masked_kup[:,12],
#              #'Putuligayuk': depth_int_ssc_summed_masked_put[:,12],
#              'Sagavanirktok': depth_int_ssc_summed_masked_sag[:,12],
#              'Staines': depth_int_ssc_summed_masked_sta[:,12],
#              'Canning': depth_int_ssc_summed_masked_can[:,12],
#              'Katakturuk': depth_int_ssc_summed_masked_kat[:,12],
#              'Hulahula': depth_int_ssc_summed_masked_hul[:,12],
#              'Jago': depth_int_ssc_summed_masked_jag[:,12],
#              #'Siksik': depth_int_ssc_summed_masked_sik[:,12],
#              'Section 1': depth_int_ssc_summed_masked_sec1[:,12],
#              'Section 2': depth_int_ssc_summed_masked_sec2[:,12],
#              'Section 3': depth_int_ssc_summed_masked_sec3[:,12],
#              'Section 4': depth_int_ssc_summed_masked_sec4[:,12],
#              'Section 5': depth_int_ssc_summed_masked_sec5[:,12],
#              'Section 6': depth_int_ssc_summed_masked_sec6[:,12],
#              'Section 7': depth_int_ssc_summed_masked_sec7[:,12],
#              'Section 8': depth_int_ssc_summed_masked_sec8[:,12],
#              'Section 9': depth_int_ssc_summed_masked_sec9[:,12],
#              'Section 10': depth_int_ssc_summed_masked_sec10[:,12],
#              'Section 11': depth_int_ssc_summed_masked_sec11[:,12],
#              'Section 12': depth_int_ssc_summed_masked_sec12[:,12], #,
#              'Section 13': depth_int_ssc_summed_masked_sec13[:,12]}

# # Make a dataframe 
# sec13_df = pd.DataFrame(sec13_data, columns=['Date']+labels_all)
# #sec1_df['Date'] = pd.to_datetime(sec1_df['Date'])


# # Make the figure
# fig13, ax13 = plt.subplots(figsize=(25,15))

# # Plot a stacked bar chart? Over time?
# # To do this, would need to cumulatively add to the bottom argument soooo this would take forever 
# # Could be worth trying to do some sort of loop?
# bottom = 0
# for l in range(len(labels_all)):
#     #print(labels[l])
#     ax13.bar(sec13_df['Date'], sec13_df[labels_all[l]], label=labels_all[l], color=colors_tmp[l])
#     bottom = bottom + sec13_df[labels_all[l]]

# plt.legend(title='Origin', bbox_to_anchor=(1.05, 1.05), loc='upper left', fontsize=fontsize, title_fontsize=fontsize, framealpha=0.5)
# ax13.set_xlabel('Date', fontsize=fontsize)
# ax13.set_ylabel('SSC (kg/m\u00b3)', fontsize=fontsize)
# ax13.set_title('Total SSC in Section 13 (kg/m\u00b3)', fontsize=fontsize)

# # Save the plot
# # Aggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Aggregated/ssc_comp_stacked_time_sec13_2020_aggregated_dbsed0001_full_0002.png', transparent=True, bbox_inches='tight', pad_inches=0)
# # Unaggregated
# #plt.savefig('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/SSC_comp_plots/Unaggregated/ssc_comp_stacked_time_sec13_2020_unaggregated_dbsed0001_full_0001.png', transparent=True, bbox_inches='tight', pad_inches=0)








# --------------------------------------------------------------------------------
# ----------------------------- Unused but Good to Have -------------------------
# --------------------------------------------------------------------------------
# =============================================================================
# # Make a pandas dataframe with the data
# sec1_depth_int_ssc_sum_comp_df = pd.DataFrame(
#      data=dict(
#          depth_int_ssc_summed_masked_kal=(['ocean_time'], depth_int_ssc_summed_masked_kal[:,0]),
#          depth_int_ssc_summed_masked_col=(['ocean_time'], depth_int_ssc_summed_masked_col[:,0]),
#          depth_int_ssc_summed_masked_sag=(['ocean_time'], depth_int_ssc_summed_masked_sag[:,0]),
#          depth_int_ssc_summed_masked_fis=(['ocean_time'], depth_int_ssc_summed_masked_fis[:,0]),
#          depth_int_ssc_summed_masked_sak=(['ocean_time'], depth_int_ssc_summed_masked_sak[:,0]),
#          depth_int_ssc_summed_masked_kup=(['ocean_time'], depth_int_ssc_summed_masked_kup[:,0]),
#          depth_int_ssc_summed_masked_put=(['ocean_time'], depth_int_ssc_summed_masked_put[:,0]),
#          depth_int_ssc_summed_masked_sta=(['ocean_time'], depth_int_ssc_summed_masked_sta[:,0]),
#          depth_int_ssc_summed_masked_can=(['ocean_time'], depth_int_ssc_summed_masked_can[:,0]),
#          depth_int_ssc_summed_masked_kat=(['ocean_time'], depth_int_ssc_summed_masked_kat[:,0]),
#          depth_int_ssc_summed_masked_hul=(['ocean_time'], depth_int_ssc_summed_masked_hul[:,0]),
#          depth_int_ssc_summed_masked_jag=(['ocean_time'], depth_int_ssc_summed_masked_jag[:,0]),
#          depth_int_ssc_summed_masked_sik=(['ocean_time'], depth_int_ssc_summed_masked_sik[:,0]),
#          depth_int_ssc_summed_masked_sec1=(['ocean_time'], depth_int_ssc_summed_masked_sec1[:,0]),
#          depth_int_ssc_summed_masked_sec2=(['ocean_time'], depth_int_ssc_summed_masked_sec2[:,0]),
#          depth_int_ssc_summed_masked_sec3=(['ocean_time'], depth_int_ssc_summed_masked_sec3[:,0]),
#          depth_int_ssc_summed_masked_sec4=(['ocean_time'], depth_int_ssc_summed_masked_sec4[:,0]),
#          depth_int_ssc_summed_masked_sec5=(['ocean_time'], depth_int_ssc_summed_masked_sec5[:,0]),
#          depth_int_ssc_summed_masked_sec6=(['ocean_time'], depth_int_ssc_summed_masked_sec6[:,0]),
#          depth_int_ssc_summed_masked_sec7=(['ocean_time'], depth_int_ssc_summed_masked_sec7[:,0]),
#          depth_int_ssc_summed_masked_sec8=(['ocean_time'], depth_int_ssc_summed_masked_sec8[:,0]),
#          depth_int_ssc_summed_masked_sec9=(['ocean_time'], depth_int_ssc_summed_masked_sec9[:,0]),
#          depth_int_ssc_summed_masked_sec10=(['ocean_time'], depth_int_ssc_summed_masked_sec10[:,0]),
#          depth_int_ssc_summed_masked_sec11=(['ocean_time'], depth_int_ssc_summed_masked_sec11[:,0]),
#          depth_int_ssc_summed_masked_sec12=(['ocean_time'], depth_int_ssc_summed_masked_sec12[:,0]),
#          depth_int_ssc_summed_masked_sec13=(['ocean_time'], depth_int_ssc_summed_masked_sec13[:,0]),
#          ocean_time=(['ocean_time'], time_steps)
#          ))
# =============================================================================
# Make a fake datetime to use 
#from datetime import datetime, timedelta
#time_tmp = np.arange(datetime(2019, 7,1, 1), datetime(2019, 8, 14, 13), timedelta(hours=4))
