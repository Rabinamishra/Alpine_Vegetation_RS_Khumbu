from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "processed"
OUTPUT_DIR = PROJECT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Read NDVI rasters
# ------------------------------------------------------------

years = [2017, 2021, 2025]

results = []

for year in years:

    raster_path = DATA_DIR / f"Gokyo_NDVI_{year}.tif"

    with rasterio.open(raster_path) as src:

        ndvi = src.read(1).astype("float32")

        nodata = src.nodata

        if nodata is not None:
            ndvi[ndvi == nodata] = np.nan

        # Keep physically meaningful NDVI values
        ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan

        valid = ndvi[np.isfinite(ndvi)]

        results.append({
            "year": year,
            "valid_pixels": len(valid),
            "mean_ndvi": np.mean(valid),
            "median_ndvi": np.median(valid),
            "std_ndvi": np.std(valid),
            "min_ndvi": np.min(valid),
            "max_ndvi": np.max(valid),
        })


# ------------------------------------------------------------
# Save statistics
# ------------------------------------------------------------

df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "ndvi_summary.csv"

df.to_csv(output_file, index=False)

print("\nNDVI SUMMARY")
print("=" * 60)
print(df.to_string(index=False))

print(f"\nSaved to:")
print(output_file)