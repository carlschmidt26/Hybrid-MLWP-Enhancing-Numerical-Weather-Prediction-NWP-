"""
This script defines the classes and methods to fetch and process weather forecast data from GFS (by NOAA/NCEI).
"""

import numpy as np
import xarray as xr
import pandas as pd

from datetime import datetime
import time

from typing import List, Dict, Any

from .weather_model_base import WeatherModelBase

class GFSModel(WeatherModelBase):
    """Class for GFS weather model.
    The Global Forecast System (GFS) is a global numerical weather prediction system run by the National Centers for Environmental Prediction (NCEP), part of NOAA.
    The dataset "NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive" is accessed via the NCAR (National Center for Atmospheric Research) THREDDS Data Server (TDS) using the OPeNDAP protocol.
    The catalog of files can be found at https://tds.gdex.ucar.edu/thredds/catalog/files/g/d084001/catalog.html.
    """
    
    def __init__(self, start_date=datetime(2015, 1, 15, 0, 0, 0), end_date=datetime(2026, 1, 30, 12, 0, 0)):
        super().__init__(model_name="GFS", base_url="https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d084001", 
                         start_date=start_date, end_date=end_date)
        
        self.grid_resolution = "0p25"       # Horizontal resolution of 0.25 degrees (~ 28 km)
        self.num_forcast_hours = [0, 3]     # Always download the +0h and +3h forecast to ensure data continuity (the +6h forecast is the +0h forecast of the next dataset).
        # [see https://github.com/aws-samples/aws-opendata-samples/blob/main/notebooks/noaa-gfs/noaa_gfs_quickstart.ipynb]

        self.dims = {
            'lat':                                      'latitude',
            'lon':                                      'longitude',
        }

        self.coords = {
            **self.dims,
            'ref_time':                                 'reference_time',               # reference time for the forecast
        }

        self.data_vars = {
            'Pressure_reduced_to_MSL_msl':              'mean_sea_level_pressure',      # mean sea level pressure
            'u-component_of_wind_height_above_ground':  'zonal_wind',                   # wind component from west to east
            'v-component_of_wind_height_above_ground':  'meridional_wind',              # wind component from south to north
        }

    def _get_run_url_pairs(self) -> List[str]:
        """Get the list of GFS forecast runs and file URLs to download data from."""
        # For each forcast run, we want to obtain two forecasts.
        forecast_runs = pd.date_range(self.start_date, self.end_date, freq='6h').repeat(2)
        date_url_pairs = [None] * len(forecast_runs)

        for idx, forecast_run in enumerate(forecast_runs):
            year_str = forecast_run.strftime('%Y')
            date_str = forecast_run.strftime('%Y%m%d')
            hour_str = forecast_run.strftime('%H')
            forecast_hour = self.num_forcast_hours[idx % 2]
            # Depending on the value of `idx`, download either the +0h and +3h forecast.
            file_url = f'{self.base_url}/{year_str}/{date_str}/gfs.{self.grid_resolution}.{date_str}{hour_str}.f{forecast_hour:03d}.grib2'
            date_url_pairs[idx] = (f'{date_str}{hour_str}_{forecast_hour:03d}', file_url)

        return date_url_pairs
    
    def _download_data_near_location(self, file_url: str, lat_target: float, lon_target: float) -> xr.Dataset:
        """Download GFS data from the given file URL and extract data at the specified location."""
        dataset = None
        download_failed_exception = None

        max_retries = 3
        try_count = 0

        while try_count < max_retries:
            try:
                # Use the thread lock to lock acces to `xr.open_dataset()`. 
                with WeatherModelBase._open_dataset_lock:
                    # Use the OPeNDAP protocol provided by the NCAR THREDDS Data Server (TDS) to fetch metadata first; only chunked data is downloaded on access.
                    # Note, that `xr.open_dataset()` might result in "RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject".
                    dataset = xr.open_dataset(file_url, engine=self.xarray_backend_engine)
                break            
            except (OSError, IOError) as e:
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str:
                    raise WeatherModelBase.RemoteFileNotFoundWarning(f'File: {file_url} not found on server.')
                else:
                    download_failed_exception = e
                
            except ValueError as e:
                raise WeatherModelBase.CorruptedFileWarning(f'File: {file_url} could not be loaded. Reason: {e}')
            
            # Simply try again for all other exceptions.
            except Exception as e:
                download_failed_exception = e

            try_count += 1
            time.sleep(2)  # Wait before retrying.

        if not dataset:
            reason = f'  Reason: {download_failed_exception}' if download_failed_exception is not None else ''
            raise WeatherModelBase.DownloadFailedException(f'File: {file_url} could not be loaded.{reason}')

        available_vars = [value for key, value in self.data_vars.items() if (key in dataset.variables)]

        if not available_vars:
            raise WeatherModelBase.CorruptedFileWarning(f'File: {file_url} does not include any of the relevant variables.')
        
        try:
            processed_dataset = self._process_and_crop_raw_data(dataset, lat_target, lon_target, available_vars)
            return processed_dataset

        except Exception as e:
            raise WeatherModelBase.DataProccesingException(f'The processing for file {file_url} failed: {e}')

        
    def _process_and_crop_raw_data(self, dataset: xr.Dataset, lat_target: float, lon_target: float, available_vars: List[str])  -> xr.Dataset:
        # In xarray, there is a distinction between a Dimension (the axis) and a Coordinate (the values/levels on which the data is stored and their unit). 
        # A single Dimension (e.g. "height") can have multiple Coordinates (the sets of values/levels) attached to it (interpreting the Dimension in different ways).
        # The primary index for label-based selection is the "Dimension Coordinate" (the Coordinate sharing the same name as the Dimension).

        # Only rename if the name actually changes to avoid ValueErrors.
        # `dataset.rename()` renames BOTH the Dimension and its matching Coordinate, keeping them linked as a primary index.
        dataset = dataset.rename({key: value for key, value in self.dims.items() if key != value})
        # Make sure that the variable also exists in the dataset, before trying to rename it.
        dataset = dataset.rename_vars({key: value for key, value in self.data_vars.items() if ((key != value) and (value in available_vars))})

        # Use the available renamed variables to filter the dataset.
        subset = dataset[available_vars]

        if 'reftime' in subset.variables:
            subset = subset.drop_vars('reftime')

        # Identify the correct dimensions that should be renamed dynamically by inspecting the 'zonal_wind' variable.
        subset = self._rename_dim_coord_dynamically(subset, 'zonal_wind', 'height')
        subset = self._rename_dim_coord_dynamically(subset, 'zonal_wind', 'time')
        
        # Get the height level closest to 10 meters above ground.
        height_levels = subset['height'].values
        height_10m = height_levels[np.abs(height_levels - 10).argmin()]

        # Get the latitude and longitude coordinates surrounding the target location.
        surrounding_grid_coords = self._get_surrounding_grid_coords(subset, lat_target, lon_target)

        # Chunk the dataset to only include the desired variables at the area around the target lat/lon.
        local_subset = subset.sel(
            latitude=slice(surrounding_grid_coords['northern_lat'], surrounding_grid_coords['southern_lat']), 
            longitude=slice(surrounding_grid_coords['western_lon'], surrounding_grid_coords['eastern_lon']),
            height=height_10m
        )

        # Since, there is only a single height level, drop the coordinate to facilitate dummy handling.
        if 'height' in subset.variables:
            subset = subset.drop_vars('height')

        # Download the data for the subset by calling .compute().
        processed_dataset =  local_subset.compute()
    
        missing_vars = [value for key, value in self.data_vars.items() if (value not in available_vars)]
        processed_dataset = self._add_missing_variables_as_nan(processed_dataset, missing_vars)

        return processed_dataset

    
    def _rename_dim_coord_dynamically(self, dataset: xr.Dataset, variable, dim_target_name) -> xr.Dataset:
        """Some "Dimension Coordinate" are not named consistently across different files. Identify the correct height dimension dynamically by inspecting the given DataArray."""
        name_candidates = [dim for dim in dataset[variable].dims if dim_target_name in dim]

        if not name_candidates:
            raise WeatherModelBase.DynamicDimensionsCoordinateRenamingException(f'Dimension Coordinate with name similar "{dim_target_name}" to not found.')
            
        name_candidate = name_candidates[0]

        # Only rename if the name actually changes to avoid ValueErrors.
        if name_candidate != dim_target_name:
            if dim_target_name in dataset.variables and dim_target_name not in dataset.dims:
                # If `dim_target_name` exists but isn't a dimension, drop it to make room for the new dimension name.
                dataset = dataset.drop_vars(dim_target_name)
            # Rename the dimension and the coordinate to the unified target name.
            dataset = dataset.rename({name_candidates[0]: dim_target_name})

        return dataset
    
    def _get_hourly_interpolated_data(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        """Interpolate data across the time dimension between datasets, where each dataset represents a single time step."""
        interpolated_datasets = [None] * (len(datasets))

        for i in range(1, len(datasets)):
            dataset1 = datasets[i - 1]
            dataset2 = datasets[i]

            time1 = dataset1['time'].values[0]
            time2 = dataset2['time'].values[0]

            time_targets = pd.date_range(time1, time2 - np.timedelta64(1, 'h'), freq='1h')

            combined_boundary = xr.concat([dataset1, dataset2], dim='time', data_vars='minimal', coords='minimal', compat='override')
            interpolated_data = combined_boundary.interp(time=time_targets, method="linear")

            interpolated_datasets[i - 1] = interpolated_data

        # Add the last dataset after the iteration.
        interpolated_datasets[-1] = dataset2

        return interpolated_datasets