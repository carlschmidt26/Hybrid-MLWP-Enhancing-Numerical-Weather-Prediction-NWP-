"""
This script defines the base class and methods to fetch and process weather forecast data from NWP models.
"""
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

import os.path
import warnings
import time

from typing import List, Dict, Tuple, Any
from tqdm import tqdm

from .weather_model_errors import  WeatherModelException, WeatherModelWarning, CorruptedFileWarning, RemoteFileNotFoundWarning, DownloadFailedException, DynamicDimensionsCoordinateRenamingException, DataProccesingException

class WeatherModelBase:
    """Base class for weather models."""

    CorruptedFileWarning = CorruptedFileWarning
    RemoteFileNotFoundWarning = RemoteFileNotFoundWarning
    DownloadFailedException = DownloadFailedException
    DynamicDimensionsCoordinateRenamingException = DynamicDimensionsCoordinateRenamingException
    DataProccesingException = DataProccesingException

    # Use a thread lock to lock to achieve thread-safety, when accessing files via `xr.open_dataset`,
    _open_dataset_lock = threading.Lock()
    
    def __init__(self, model_name: str, base_url: str, start_date: datetime, end_date: datetime, data_path: str = "./raw_data/"):
        self.processing_summary_verbosity_limit = None

        self.model_name = model_name
        self.base_url = base_url
        self.data_path = os.path.join(data_path, model_name)
        
        self.start_date = start_date
        self.end_date = end_date

        # The "engine" provides a set of instructions that tells Xarray how to read the data and pack them into a DataSet (or DataArray). 
        # These instructions are stored in an underlying "backend" [see https://docs.xarray.dev/en/stable/user-guide/io.html].
        # The "netcdf4" engine uses Hierarchical Data Format 5 (HDF5) internally, which is not thread-safe in many installations. Use "pydap" insead.
        self.xarray_backend_engine = "netcdf4"  # Default backend engine for xarray.

        # Create the data directory if necessary.
        os.makedirs(self.data_path, exist_ok=True)

    def _get_run_url_pairs(self) -> List[str]:
        """Get the list of forecast runs and file URLs to download data from."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _download_data_near_location(self, file_url: str, lat_target: float, lon_target: float) -> xr.Dataset:
        """Download data from the given file URL and extract data at the specified location."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _get_hourly_interpolated_data(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        """Download data from the given file URL and extract data at the specified location."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_data_at_location(self, lat_target: float, lon_target: float, download=False, max_workers=5) -> List[xr.Dataset]:
        """Fetch data for the given latitude and longitude."""
        forecast_run_url_pairs = self._get_run_url_pairs()

        with warnings.catch_warnings(record=True, action='always') as caught_warnings:
            # Ignore warnings that might be raised when calling `xr.open_dataset()`.
            warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

            # Instead of an explicit for loop, use `concurrent.futures.ThreadPoolExecutor.map()` to create a list of the individual datasets.
            # [see https://docs.python.org/3/library/concurrent.futures.html#:~:text=map(fn%2C%20*iterables%2C%20timeout%3DNone%2C%20chunksize%3D1%2C%20buffersize%3DNone)]
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                result_iterator = tqdm(
                    executor.map(lambda forecast_run_url_pair: self._process_single_file(forecast_run_url_pair, lat_target, lon_target, download), forecast_run_url_pairs), 
                    total=len(forecast_run_url_pairs)
                )

                datasets = list(result_iterator)

        self._print_processing_summary(caught_warnings)

        if not datasets:
            return None
        elif len(datasets) == 1:
            return datasets[0]

        hourly_resolved_datasets = self._get_hourly_interpolated_data(datasets)

        full_dataset = xr.concat(hourly_resolved_datasets, dim='time', data_vars='minimal', coords='minimal', compat='override')

        return full_dataset
    
    def _process_single_file(self, forecast_run_url_pair: tuple, lat_target: float, lon_target: float, download: bool):
        forecast_run, file_url = forecast_run_url_pair
        full_file_path = os.path.join(self.data_path, f'{self.model_name}_{forecast_run}.nc')
        # Check if the file has already been downloaded. If not, download it.
        # (Note, that the file is downloaded anyway, if download=True).
        file_exists = os.path.exists(full_file_path)

        if file_exists and not download:
            # With `with` here to ensure that the dataset is properly closed after processing
            with xr.open_dataset(full_file_path) as interpolated_dataset:
                return interpolated_dataset.load()
        else:
            try:
                # Note, that this will at most return data for the area surrounding the target lat/lon (i.e., four data points per variable).
                local_dataset = self._download_data_near_location(file_url, lat_target, lon_target)

                interpolated_dataset = local_dataset.interp(latitude=lat_target, longitude=lon_target, method="linear")

                # Save the interpolated file to disk.
                interpolated_dataset.to_netcdf(full_file_path)

            # Catch all relevant exceptions and warnings, that were raised while downloading the data.
            except (WeatherModelException, WeatherModelWarning) as e:
                if isinstance(e, WeatherModelException):
                    self._handle_weather_model_exceptions(e)
                if isinstance(e, WeatherModelWarning):
                    self._handle_weather_model_warnings(e)

                return self._create_dummy_dataset(lat_target, lon_target, forecast_run)
        
        return interpolated_dataset

    def _get_surrounding_grid_coords(self, dataset: xr.Dataset, lat_target: float, lon_target: float) -> Dict[str, Any]:
        """Get the four surrounding grid points for a given latitude and longitude."""
        lats = dataset['latitude'].values
        lons = dataset['longitude'].values

        northern_lat = lats[lats >= lat_target].min()
        southern_lat = lats[lats <= lat_target].max()
        eastern_lon = lons[lons >= lon_target].min()
        western_lon = lons[lons <= lon_target].max()

        grid_coords = {
            'northern_lat': northern_lat,
            'southern_lat': southern_lat,
            'eastern_lon': eastern_lon,
            'western_lon': western_lon
        }

        return grid_coords
    
    def _print_processing_summary(self, caught_warnings: List[Warning]):
        """Print all collected warnings (depending on the filters set)"""
        if not caught_warnings:
            print('✅ All files have been processed successfully.')

        missing_file_warnings = [w for w in caught_warnings if issubclass(w.category, WeatherModelBase.RemoteFileNotFoundWarning)]
        corrupted_file_warnings = [w for w in caught_warnings if issubclass(w.category, WeatherModelBase.CorruptedFileWarning)]
        other_warnings = [w for w in caught_warnings if (not issubclass(w.category, WeatherModelBase.RemoteFileNotFoundWarning)) and (not issubclass(w.category, WeatherModelBase.CorruptedFileWarning))]

        if missing_file_warnings:
            print(f'⚠️  {len(missing_file_warnings)} files could not be read from the server.')
            for missing_file_warning in missing_file_warnings[:self.processing_summary_verbosity_limit]:
                print(f'   - {missing_file_warning.message}')
            if (self.processing_summary_verbosity_limit is not None) and (len(missing_file_warnings) > self.processing_summary_verbosity_limit):
                print(f'   ... and {len(missing_file_warnings)-self.processing_summary_verbosity_limit} additional files.')
            
        if corrupted_file_warnings:
            print(f'❌ {len(corrupted_file_warnings)} files were corrupted/could not be loaded.')
            for corrupted_file_warning in corrupted_file_warnings[:self.processing_summary_verbosity_limit]:
                print(f'   - {corrupted_file_warning.message}')
            if (self.processing_summary_verbosity_limit is not None) and (len(corrupted_file_warnings) > self.processing_summary_verbosity_limit):
                print(f'   ... and {len(corrupted_file_warnings)-self.processing_summary_verbosity_limit} additional files.')

        if other_warnings:
            print(f'ℹ️  {len(other_warnings)} other issues occurred during processing:')
            for warning in other_warnings:
                print(f'   - {warning.message}')

    def _handle_weather_model_exceptions(self, error: WeatherModelException):
        # Use `stacklevel=2` to point to the method calling `self._handle_weather_model_exceptions()`.
        warnings.warn(str(error), stacklevel=2)

    def _handle_weather_model_warnings(self, warning: WeatherModelWarning):
        # Could use `type(warning)` equivalently here.
        category = warning.__class__
        # Use `stacklevel=2` to point to the method calling `self._handle_weather_model_exceptions()`.
        warnings.warn(str(warning), category=category, stacklevel=2)


    def _create_dummy_dataset(self, lat_target: float, lon_target: float, forecast_run: str):
        """Create a dummy dataset for data that is not available."""
        base_data_str, forecast_hour_str = forecast_run.split('_')

        coords = {
            'latitude': np.array(lat_target),
            'longitude': np.array(lon_target),
            'time': np.array([datetime.strptime(base_data_str, '%Y%m%d%H') + timedelta(hours=int(forecast_hour_str))])
        }
        # Data variables can be defined via tuples of the form (dims, data[, attrs]).
        data_vars = {var: (['time'], np.array([np.nan]), {'units': 'missing'}) for var in self.data_vars.values()}

        dummy_dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=dict(description="Dummy dataset for corrupted/unavailable data."),
        )

        return dummy_dataset
    
    def _add_missing_variables_as_nan(self, dataset: xr.Dataset, missing_variables: List[str]) -> xr.Dataset:
        dimensions = dataset.dims
        for missing_var in missing_variables:
            # Data variables can be defined via tuples of the form (dims, data[, attrs]).
            dataset[missing_var] = (list(dimensions), np.array([np.nan]).reshape([1] * len(dimensions)), {'units': 'missing'})
        
        return dataset