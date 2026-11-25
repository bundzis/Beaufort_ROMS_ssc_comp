# <span style="color:skyblue"> Suspended Sediment Compositions on the Alaskan Beaufort Sea Shelf During the 2020 Open Water Season </span> 
Thie repo *(Beufort_ROMS_ssc_comp)* is home to the analysis scripts used for the project about 
the composition of SSC on the Alaskan Beaufort Sea shelf in terms
of riverine sediment, local resuspension, and regional advection.
This repo also contains the settings/scripts needed to run 
ROMS.

The associated ROMS input files will be uploaded to a Zenodo and linked 
here later.

Note: For some analysis scripts, the *.py version is for processing 
the model output (needed highmem nodes) and the *.ipynb version is 
for plotting the post-processed data. 

## <span style="color:coral"> *Running ROMS* </span>
<ins> **Unique Aggregated Run** </ins>
- build script: /Model_setup/Unique_aggregated/coawst.bash
- Header file: /Model_setup/Unique_aggregated/beaufort_rivers_ak_002.h
- ocean.in: /Model_setup/Unique_aggregated/ocean_beaufort_ak_002.in
- sediment.in: /Model_setup/Unique_aggregated/sediment_beaufort_ak_002_10rivs.in
- run script: /Model_setup/Unique_aggregated/run_alpine  

<ins> **Unique Unaggregated Run** </ins>
- build script: /Model_setup/Unique_unaggregated/coawst.bash
- header file: /Model_setup/Unique_unaggregated/beaufort_rivers_ak_003.h
- ocean.in: /Model_setup/Unique_unaggregated/ocean_beaufort_ak_003.in
- sediment.in: /Model_setup/Unique_unaggregated/sediment_beaufort_ak_003_10rivs.in
- run script: /Model_setup/Unique_unaggregated/run_alpine

<ins> **Double Discharge Run** </ins>
  - ocean.in: /Model_setup/Double_dis/ocean_beaufort_ak_002_double_dis.in
  - sediment.in: /Model_setup/Double_dis/sediment_beaufort_ak_002_10rivs_riv_sensitivity.in
  - run script: /Model_setup/Double_dis/run_alpine_double

Variable info: varinfo.dat

## <span style="color:violet"> *Analysis* </span>
- Connectiviy matrices
  - <ins>Script:</ins> connectivity_matrix.ipynb 
  - Figure 8
- Initial conditions (unique unaggregated run)
  - <ins>Script:</ins> initial_conds_beaufort_shelf_rivs_003_2020_Blaskey.ipynb
  - Figure 3
- Calculate percentage of different sediment classes in SSC over time
  - <ins>Scripts:</ins> percent_ssc_diff_sed_classes.py, percent_ssc_diff_sed_classes.ipynb, run_percent_ssc_diff_sed_classes
  - Post-processed data: timeseries_ssc_by_class_agg_run.nc, timeseries_ssc_by_class_unag_run.nc
- Box & whisker plots of percent of SSC that is locally resuspended
  - <ins>Scripts:</ins> plot_box_whiskers_local_resuspend_over_time.py, plot_box_whiskers_local_resuspend_over_time.ipynb, run_plot_box_whiskers_local_resuspend_over_time
  - Figure 7
  - Post-processed data: percent_local_resuspended_over_time_aggregated.nc, percent_local_resuspended_over_time_unaggregated.nc
- Box & whisker plots of percent riverine sed in diff depths
  - <ins>Scripts:</ins> plot_box_whiskers_river_sed_percent_time.py, plot_box_whiskers_river_sed_percent_time.ipynb, run_plot_box_whiskers_river_sed_percent_time, plot_box_whiskers_river_sed_percent_time_sensitivity_test.py, plot_box_whiskers_river_sed_percent_time_sensitivity_test.ipynb, run_plot_box_whiskers_river_sed_percent_time_sensitivity_test
  - Figure 11
  - Post-processed data: percent_river_suspended_sed_over_time_in_regions_aggregated.nc, percent_river_suspended_sed_over_time_in_regions_unaggregated.nc
- Model grid
  - <ins>Script:</ins> plot_grid_figure_paper2.ipynb
  - Figure 1
- Model forcing timeseries
  - <ins>Script:</ins> plot_model_forcing_time_series_2020_paper2.py
  - Figure 2
- Model-observatio comparison
  - <ins>Script:</ins> plot_eidam_roms_ssc_maps_paper2.py
  - Figure 4
- Percent SSC riv vs. seabed
  - <ins>Scripts:</ins> plot_percent_ssc_rivs_seabed.py, plot_percent_ssc_rivs_seabed.ipynb, run_plot_percent_ssc_rivs_seabed
  - Figure 12
  - Post-processed data: timeseries_ssc_seabed_river_std_runs.nc, timeseries_ssc_seabed_river_doub_dis_run.nc
- Percent time SSC above thresholds
  - <ins>Scripts:</ins> plot_percent_time_ssc_above_value.py, plot_percent_time_ssc_above_value.ipynb, run_plot_percent_time_ssc_above_value
  - Figure 5
  - Post-processed data: time_ssc_top_1m_above_three_thresholds_agg.nc, time_ssc_top_1m_above_three_thresholds_unag.nc
- River kg in regions over time
  - <ins>Scripts:</ins> plot_river_sed_kg_regions_time.py, plot_river_sed_kg_regions_time.ipynb, run_plot_river_sed_kg_regions_time
  - Post-processed data: kg_suspended_sed_from_rivers_in_depth_regions_aggregated_dbsed0001.nc, kg_suspended_sed_from_rivers_in_depth_regions_unaggregated_dbsed0001.nc
- Stacked barcharts SSC composition over time
   - <ins>Scripts:</ins> plot_ssc_composition_in_sections_barcharts_over_time.py, plot_ssc_composition_in_sections_barcharts_over_time.ipynb, run_plot_ssc_composition_in_sections_barcharts_over_time
  - Figure 9
  - Post-processed data: percent_river_suspended_sed_over_time_in_regions_aggregated.nc, depth_int_ssc_summed_from_source_in_sections_aggregated_dbsed0001.nc, depth_int_ssc_summed_from_source_in_sections_unaggregated_dbsed0001.nc
- Movie and plot of surface salinity, stress, and riverine SSC snapshots
   - <ins>Script:</ins> plot_surf_salt_stress_movies.ipynb
  - Figure 10
- Spatially-averaged waves, bottom cur. mag., SSC by section
   - <ins>Scripts:</ins> plot_time_series_spatial_avg_ssc_waves_curs_bstr_by_section.py, plot_time_series_spatial_avg_ssc_waves_curs_bstr_by_section.ipynb, run_plot_time_series_spatial_avg_ssc_waves_curs_bstr_by_section
  - Post-processed data: timeseries_ssc_hwave_curmag_bsrt_by_section_agg.nc, timeseries_ssc_hwave_curmag_bsrt_by_section_unag.nc, timeseries_ssc_hwave_curmag_bsrt_by_section_agg_nosec13.nc, timeseries_ssc_hwave_curmag_bsrt_by_section_unag_nosec13.nc
- Maps of correlations
   - <ins>Script:</ins> plot_correlations_as_maps.ipynb
  - Figure 6
