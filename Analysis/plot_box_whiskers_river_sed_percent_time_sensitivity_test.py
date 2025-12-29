#################### Plot Box and Whisker Plots of Percent of ###################################
############# Suspended Sediment that is Locally Suspended Over Time ############################
######################### With Sensitivity Test(s) ##############################################
# The purpose of this script is to make plots that have one box and whisker plot for 
# each shelf section that has the range of percentages of locally resuspended 
# sediment in that section over time. This is the same as plot_box_whiskers_river_sed_percent_time.py
# but edited to compared the standard runs to the sensitivity test(s) with different river
# discharge and sediment loads.
#
# Notes:
# - This leaves out the rivers that are no longer in the 2020 model
#   runs that use Blaskey river data 
# - This has been edited to read in the pre-processed data for standard runs and 
#   process the sensitivity test data 
# - This is a copy of the *.ipynb version made solely for processing the standard runs
#########################################################################################

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
import seaborn as sns
import matplotlib.patches as mpatches


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


# ------------------------------------------------------------------------------------
# ------------------------- Define a bunch of functions ------------------------------
# ------------------------------------------------------------------------------------

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


    # Make a function that is the same as above but adds together 
# all riverine sediment to get bulk river values for standard runs 
# Make a function to get the depth-integrated kg sediment per cell in space
# as a time series
def get_depth_int_ssc_timeseries_bulk_river(filename):
    """
    This function takes a model output file, pulls out the desired 
    sediment class, gets the depth-integrated SSC as a time series.

    Parameters
    ----------
    filename : The name/path of the model output file

    Returns
    -------
    None.

    """
    
    # Load in the model output
    model_output = xr.open_dataset(filename)

    # Add together all the riverine sediment classes
    ssc_all_riv = (model_output.mud_15 + model_output.mud_16 + model_output.mud_17 + model_output.mud_18 +
                  model_output.mud_19 + model_output.mud_20 + model_output.mud_21 + model_output.mud_22 +
                  model_output.mud_23 + model_output.mud_24)
    
    # To collapse to horizontal, multiply each layer by its
    # thickness
    # Calculate the time-varying thickness of the cells
    dz = abs(model_output.z_w[:,:-1,:,:].values - model_output.z_w[:,1:,:,:].values)
    
    # Multiply the SSC by thick thickness
    depth_int_ssc_all_riv_tmp = (ssc_all_riv*dz)
    
    # Then depth-integrated by summing over depth and dividing by dy
    depth_int_ssc_all_riv = (depth_int_ssc_all_riv_tmp.sum(dim='s_rho'))
    
    # Now we have SSC in kg/m2 for each grid cell in horizontal
    # Return this value - can multiply by dx and dy outside the function
    return(depth_int_ssc_all_riv)


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


# ---- Percent Over Time ----
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



# ------------------------------------------------------------------------------------
# ------------------------- Make the Masks for Regions ------------------------------
# ------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------
# ------------------------- Process the Standard Runs ------------------------------
# ------------------------------------------------------------------------------------

# Load in the standard run output
# -- Aggregated --
# 2020 dbsed0001 - full run 
#file_names = glob('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Aggregated/dbsed0001_10rivs/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
file_names_agg = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_002_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_aggregated_dbsed0001_*.nc')
# -- Unaggregated --
# 2020 dbsed0001 - full run
#file_names = glob('/pl/active/moriarty_lab/BriannaU/Paper2/Model_Output/Unaggregated/dbsed0001_10rivs_unaggregated/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc') 
file_names_unag = glob('/scratch/alpine/brun1463/ROMS_scratch/Beaufort_Shelf_Rivers_Alpine_003_scratch/ocean_his_beaufort_rivers_10rivs_13seabed_unaggregated_dbsed0001_*.nc')

# Sort them to be in order
# Aggregated 
file_names2_agg = sorted(file_names_agg)

# Check to see if this worked
print(file_names2_agg[0], flush=True)
print(file_names2_agg[1], flush=True)
print(file_names2_agg[2], flush=True)
print(file_names2_agg[-1], flush=True)
print('all files: ', file_names2_agg, flush=True)

# Pull out the number of files
num_files_agg = len(file_names2_agg)

# Unaggregated
file_names2_unag = sorted(file_names_unag)

# Check to see if this worked
print(file_names2_unag[0], flush=True)
print(file_names2_unag[1], flush=True)
print(file_names2_unag[2], flush=True)
print(file_names2_unag[-1], flush=True)
print('all files: ', file_names2_unag, flush=True)

# Pull out the number of files
num_files_unag = len(file_names2_unag)

print(file_names2_agg, flush=True)
print(file_names2_unag, flush=True)

# Pull out the length of time of the full run, the time steps, 
# and the length of time of each output file
full_time_len_agg, time_steps_agg, time_lengths_agg = get_model_time(file_names2_agg, num_files_agg)
full_time_len_unag, time_steps_unag, time_lengths_unag = get_model_time(file_names2_unag, num_files_unag)

# Make some arrays to hold output
# One for each sediment type in the river muds - look at guide
# Unagggregated
depth_int_ssc_unag_std = np.empty((full_time_len_unag, eta_rho_len, xi_rho_len))
# Aggregated 
depth_int_ssc_agg_std = np.empty((full_time_len_agg, eta_rho_len, xi_rho_len))

# Set a time step to track which time step the loop is on
time_step_agg = 0
time_step_unag = 0

# Loop through the model output
for j in range(num_files_agg):
#for j in range(1):

    print('j: ', j, flush=True)
    
    # Call the function to process the output and Save these to the arrays 
    #print('time_step: ', time_step)
    #print('time_step + time_lengths[j]: ', time_step+time_lengths[j])
    start_agg = int(time_step_agg)
    end_agg = int(time_step_agg+time_lengths_agg[j])
    start_unag = int(time_step_unag)
    end_unag = int(time_step_agg+time_lengths_unag[j])
    
    # Unaggregated
    depth_int_ssc_unag_std[start_unag:end_unag,:,:] = get_depth_int_ssc_timeseries_bulk_river(file_names2_unag[j])
    # Aggregated
    depth_int_ssc_agg_std[start_agg:end_agg,:,:] = get_depth_int_ssc_timeseries_bulk_river(file_names2_agg[j])

    # Update the base time_step
    time_step_agg = time_step_agg + time_lengths_agg[j]
    time_step_unag = time_step_unag + time_lengths_unag[j]

# Okay now multiply by dx and dy, and rename to be by river and combine where needed
dx = 750 # meters
dy = 600 # meters 

# Unaggregated
kg_sed_riv_unag_std = depth_int_ssc_unag_std*dx*dy # kg
# Aggregated
kg_sed_riv_agg_std = depth_int_ssc_agg_std*dx*dy # kg

# Check the shapes 
print(np.shape(kg_sed_riv_unag_std), flush=True)

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
kg_sed_riv_unag_std_masked = np.empty((full_time_len_unag,eta_rho_len,xi_rho_len))
kg_sed_riv_agg_std_masked = np.empty((full_time_len_agg,eta_rho_len,xi_rho_len))

# Loop through time
# Aggregated
for t in range(full_time_len_agg):
    print('t: ', t, flush=True) 
    # Mask
    kg_sed_riv_agg_std_masked[t,:,:] = kg_sed_riv_agg_std[t,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan
# Unaggregated
for tt in range(full_time_len_unag):
    print('tt: ', tt, flush=True)
    
    # Mask
    kg_sed_riv_unag_std_masked[tt,:,:] = kg_sed_riv_unag_std[tt,:,:]*temp_mask*mask_rho_nan.nudge_mask_rho_nan


# Now multiply by the mask to get the different regions 
# Make empty arrays to hold the values
# Unaggregated
kg_sed_0_10m_unag_std = np.empty((full_time_len_unag,eta_rho_len,xi_rho_len))
kg_sed_10_20m_unag_std = np.empty((full_time_len_unag,eta_rho_len,xi_rho_len))
kg_sed_20_30m_unag_std = np.empty((full_time_len_unag,eta_rho_len,xi_rho_len))
kg_sed_30_60m_unag_std = np.empty((full_time_len_unag,eta_rho_len,xi_rho_len))
# Aggregated
kg_sed_0_10m_agg_std = np.empty((full_time_len_agg,eta_rho_len,xi_rho_len))
kg_sed_10_20m_agg_std = np.empty((full_time_len_agg,eta_rho_len,xi_rho_len))
kg_sed_20_30m_agg_std = np.empty((full_time_len_agg,eta_rho_len,xi_rho_len))
kg_sed_30_60m_agg_std = np.empty((full_time_len_agg,eta_rho_len,xi_rho_len))

# Loop through time 
# Aggregated
for tt in range(full_time_len_agg):
    print('tt: ', tt, flush=True)
    # Aggregated
    kg_sed_0_10m_agg_std[tt,:,:] = kg_sed_riv_agg_std_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_agg_std[tt,:,:] = kg_sed_riv_agg_std_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_agg_std[tt,:,:] = kg_sed_riv_agg_std_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_agg_std[tt,:,:] = kg_sed_riv_agg_std_masked[tt,:,:]*outer_30_60m_mask_rho_nan
# Unaggregated
for tt in range(full_time_len_unag):
    print('tt: ', tt, flush=True)
        
    # Unaggregated
    kg_sed_0_10m_unag_std[tt,:,:] = kg_sed_riv_unag_std_masked[tt,:,:]*inner_10m_mask_rho_nan
    kg_sed_10_20m_unag_std[tt,:,:] = kg_sed_riv_unag_std_masked[tt,:,:]*outer_10_20m_mask_rho_nan
    kg_sed_20_30m_unag_std[tt,:,:] = kg_sed_riv_unag_std_masked[tt,:,:]*outer_20_30m_mask_rho_nan
    kg_sed_30_60m_unag_std[tt,:,:] = kg_sed_riv_unag_std_masked[tt,:,:]*outer_30_60m_mask_rho_nan


# Trim
# Unaggregated
kg_sed_0_10m_unag_std_masked_trimmed = kg_sed_0_10m_unag_std[:,:,c_west:-c_west]
kg_sed_10_20m_unag_std_masked_trimmed = kg_sed_10_20m_unag_std[:,:,c_west:-c_west]
kg_sed_20_30m_unag_std_masked_trimmed = kg_sed_20_30m_unag_std[:,:,c_west:-c_west]
kg_sed_30_60m_unag_std_masked_trimmed = kg_sed_30_60m_unag_std[:,:,c_west:-c_west]
# Aggregated
kg_sed_0_10m_agg_std_masked_trimmed = kg_sed_0_10m_agg_std[:,:,c_west:-c_west]
kg_sed_10_20m_agg_std_masked_trimmed = kg_sed_10_20m_agg_std[:,:,c_west:-c_west]
kg_sed_20_30m_agg_std_masked_trimmed = kg_sed_20_30m_agg_std[:,:,c_west:-c_west]
kg_sed_30_60m_agg_std_masked_trimmed = kg_sed_30_60m_agg_std[:,:,c_west:-c_west]

# Then assuming it worked, sum over each region to get a kg per time for each
# class in each region
# Unaggregated 
kg_sed_0_10m_unag_std_masked_trimmed_sum = np.nansum(kg_sed_0_10m_unag_std_masked_trimmed, axis=(1,2))
kg_sed_10_20m_unag_std_masked_trimmed_sum = np.nansum(kg_sed_10_20m_unag_std_masked_trimmed, axis=(1,2))
kg_sed_20_30m_unag_std_masked_trimmed_sum = np.nansum(kg_sed_20_30m_unag_std_masked_trimmed, axis=(1,2))
kg_sed_30_60m_unag_std_masked_trimmed_sum = np.nansum(kg_sed_30_60m_unag_std_masked_trimmed, axis=(1,2))
# Aggregated 
kg_sed_0_10m_agg_std_masked_trimmed_sum = np.nansum(kg_sed_0_10m_agg_std_masked_trimmed, axis=(1,2))
kg_sed_10_20m_agg_std_masked_trimmed_sum = np.nansum(kg_sed_10_20m_agg_std_masked_trimmed, axis=(1,2))
kg_sed_20_30m_agg_std_masked_trimmed_sum = np.nansum(kg_sed_20_30m_agg_std_masked_trimmed, axis=(1,2))
kg_sed_30_60m_agg_std_masked_trimmed_sum = np.nansum(kg_sed_30_60m_agg_std_masked_trimmed, axis=(1,2))
# Combined aggregated and unaggregated 
kg_sed_0_10m_all_std_masked_trimmed_sum = kg_sed_0_10m_unag_std_masked_trimmed_sum[:738] + kg_sed_0_10m_agg_std_masked_trimmed_sum[:738]
kg_sed_10_20m_all_std_masked_trimmed_sum = kg_sed_10_20m_unag_std_masked_trimmed_sum[:738] + kg_sed_10_20m_agg_std_masked_trimmed_sum[:738]
kg_sed_20_30m_all_std_masked_trimmed_sum = kg_sed_20_30m_unag_std_masked_trimmed_sum[:738] + kg_sed_20_30m_agg_std_masked_trimmed_sum[:738]
kg_sed_30_60m_all_std_masked_trimmed_sum = kg_sed_30_60m_unag_std_masked_trimmed_sum[:738] + kg_sed_30_60m_agg_std_masked_trimmed_sum[:738]

# Call the function for each river 
# Unaggregated
percent_0_10m_unag_std, percent_10_20m_unag_std, percent_20_30m_unag_std, percent_30_60m_unag_std = get_percent_regions_over_time_by_river(kg_sed_0_10m_unag_std_masked_trimmed_sum, kg_sed_10_20m_unag_std_masked_trimmed_sum, kg_sed_20_30m_unag_std_masked_trimmed_sum, kg_sed_30_60m_unag_std_masked_trimmed_sum)
# Aggregated
percent_0_10m_agg_std, percent_10_20m_agg_std, percent_20_30m_agg_std, percent_30_60m_agg_std = get_percent_regions_over_time_by_river(kg_sed_0_10m_agg_std_masked_trimmed_sum, kg_sed_10_20m_agg_std_masked_trimmed_sum, kg_sed_20_30m_agg_std_masked_trimmed_sum, kg_sed_30_60m_agg_std_masked_trimmed_sum)
# Combined aggregated and unaggregated 
percent_0_10m_all_std, percent_10_20m_all_std, percent_20_30m_all_std, percent_30_60m_all_std = get_percent_regions_over_time_by_river(kg_sed_0_10m_all_std_masked_trimmed_sum, kg_sed_10_20m_all_std_masked_trimmed_sum, kg_sed_20_30m_all_std_masked_trimmed_sum, kg_sed_30_60m_all_std_masked_trimmed_sum)


# ---------------- Trim to Time we Trust --------------------
# Trim all of these variables to the time that we trust 
# Unaggregated 
percent_0_10m_unag_std = percent_0_10m_unag_std[:738]
percent_10_20m_unag_std = percent_10_20m_unag_std[:738]
percent_20_30m_unag_std = percent_20_30m_unag_std[:738]
percent_30_60m_unag_std = percent_30_60m_unag_std[:738]
# Aggregated
percent_0_10m_agg_std = percent_0_10m_agg_std[:738]
percent_10_20m_agg_std = percent_10_20m_agg_std[:738]
percent_20_30m_agg_std = percent_20_30m_agg_std[:738]
percent_30_60m_agg_std = percent_30_60m_agg_std[:738]
# Combined aggregated and unaggregated 
# percent_0_10m_all_std = percent_0_10m_all_std[:738]
# percent_10_20m_all_std = percent_10_20m_all_std[:738]
# percent_20_30m_all_std = percent_20_30m_all_std[:738]
# percent_30_60m_all_std = percent_30_60m_all_std[:738]

# Save to a netcdf
# Save these to a netcdf since the analysis takes so long?
percent_regions_over_time_bulk_std = xr.Dataset(
    data_vars=dict(
        percent_0_10m_unag_std=(['ocean_time'], percent_0_10m_unag_std),
        percent_10_20m_unag_std=(['ocean_time'], percent_10_20m_unag_std),
        percent_20_30m_unag_std=(['ocean_time'], percent_20_30m_unag_std),
        percent_30_60m_unag_std=(['ocean_time'], percent_30_60m_unag_std),
        percent_0_10m_agg_std=(['ocean_time'], percent_0_10m_agg_std),
        percent_10_20m_agg_std=(['ocean_time'], percent_10_20m_agg_std),
        percent_20_30m_agg_std=(['ocean_time'], percent_20_30m_agg_std),
        percent_30_60m_agg_std=(['ocean_time'], percent_30_60m_agg_std),
        percent_0_10m_all_std=(['ocean_time'], percent_0_10m_all_std),
        percent_10_20m_all_std=(['ocean_time'], percent_10_20m_all_std),
        percent_20_30m_all_std=(['ocean_time'], percent_20_30m_all_std),
        percent_30_60m_all_std=(['ocean_time'], percent_30_60m_all_std),
    ),
    coords=dict(
        ocean_time=('ocean_time', time_steps_agg[:738])
    ),
    attrs=dict(description='Time-series ROMS output of percent of riverine suspended sediment in each region by sediment class for the standard runs')
)

# Save to a netcdf
# Standard
percent_regions_over_time_bulk_std.to_netcdf('/projects/brun1463/ROMS/Beaufort_Shelf_Rivers_Alpine_002/Scripts/Analysis/Percent_river_suspended_sed_regions/percent_river_suspended_sed_over_time_in_regions_bulk_std_03.nc')

