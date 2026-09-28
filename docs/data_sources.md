# Data Sources

## Sentinel-2

Sentinel-2 Level-2A Surface Reflectance data were accessed through Google Earth Engine using the `COPERNICUS/S2_SR_HARMONIZED` collection.

### Processing
- Analysis period: 2017–2025
- Comparable seasonal window: 1 September–1 November
- Scene-level cloud filter: <60% cloudy pixels
- Pixel-level masking: Sentinel-2 Scene Classification Layer (SCL)
- Retained SCL classes: 2, 4, 5, and 6
- NDVI: (NIR - Red) / (NIR + Red)
- Red band: B4
- Near-infrared band: B8
- Output resolution: 10 m
- Annual composites: median of the available valid observations

The 2017 period contained fewer usable observations than 2021 and 2025. This is considered a limitation of the temporal comparison.

## Digital Elevation Model

Copernicus DEM GLO-30 was accessed through Google Earth Engine.

The DEM was used to derive:
- Elevation
- Slope
- Aspect

## Study Area

The analysis uses a compact research area in the Gokyo Valley, Khumbu Himal, Nepal.

Approximate study-area extent:

- Longitude: 86.65–86.74° E
- Latitude: 27.92–28.02° N

The study area is a research window and is not intended to represent an official administrative or protected-area boundary.

## Data Availability

Large Sentinel-2 and DEM raster files are not included in the GitHub repository.

The repository contains:
- Analysis scripts
- Derived summary tables
- Figures
- Data-source documentation
- Reproducibility information

The original raster datasets can be regenerated from the documented data sources and processing workflow.