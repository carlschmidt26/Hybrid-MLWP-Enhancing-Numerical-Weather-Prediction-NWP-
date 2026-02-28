# Dataset Characteristics

**[Notebook](exploratory_data_analysis.ipynb)**

## Dataset Information

### Dataset Source
- **Dataset Link:** [NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive](https://gdex.ucar.edu/datasets/d084001/). Data has been obtained via [https://tds.gdex.ucar.edu/thredds/catalog/files/g/d084001/catalog.html](https://tds.gdex.ucar.edu/thredds/catalog/files/g/d084001/catalog.html).
- **Dataset Owner/Contact:** DOC/NOAA/NWS/NCEP > National Centers for Environmental Prediction, National Weather Service, NOAA, U.S. Department of Commerce. 2015, updated daily. NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive. NSF National Center for Atmospheric Research. https://doi.org/10.5065/D65D8PWK.
### Dataset Characteristics
- **Number of Observations:** In total, 32,162 individual files were downloaded and linearly interpolated in the temporal dimension from the original 3-hourly interval to an hourly resolution. The individual files were subsequently merged into a single file.
- **Number of Features:** Three primary physical variables: *u-component of wind* (zonal wind) [m/s], *v-component of wind* (meridional wind) [m/s], and *mean sea level pressure* [Pa].
### Target Variable/Label
- **Label Name:** *u-component of wind* [m/s] and *v-component of wind* [m/s]
- **Label Type:** Regression (Next Token Prediction).
- **Label Description:** The Next Token Prediction uses an input sequence of the stated variables and predicts the next value that would follow the most recent value in the input sequence
- **Label Values:** Floating point numbers, typically within a range of ±20 m/s.
- **Label Distribution:** The distributions of the target values roughly follow a normal distribution. While the mean for the meridional wind is approximately centered around 0 m/s, there is a slight bias towards positive values for the zonal wind, indicating the prevalence of westerly winds at the location at hand.

### Feature Description

- **Feature 1 (u-component of wind):** West-east wind component [m/s]. Positive values indicate eastward (westerly) flow.
- **Feature 2 (v-component of wind):** North-south wind component [m/s]. Positive values indicate northward (southerly) flow.
- **Feature 3 (mean_sea_level_pressure):** Air pressure reduced to mean sea level [Pa].

## Exploratory Data Analysis

The exploratory data analysis is conducted in the [exploratory_data_analysis.ipynb](exploratory_data_analysis.ipynb) notebook, which includes:
- Data loading and initial inspection
- Statistical summaries and distributions
- Missing value analysis
- Feature correlation analysis (focusing on auto-correlations)
- Data visualization and insights
- Data quality assessment
