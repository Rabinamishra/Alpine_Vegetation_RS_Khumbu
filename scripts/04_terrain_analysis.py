from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

dem_path = DATA_DIR / "Gokyo_DEM.tif"
ndvi_path = DATA_DIR / "Gokyo_NDVI_2025.tif"


# ------------------------------------------------------------
# Read NDVI
# ------------------------------------------------------------

with rasterio.open(ndvi_path) as src:

    ndvi = src.read(1).astype("float32")

    ndvi_profile = src.profile
    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_shape = src.shape


# Remove invalid NDVI values
ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan


# ------------------------------------------------------------
# Read DEM
# ------------------------------------------------------------

with rasterio.open(dem_path) as src:

    dem = src.read(1).astype("float32")

    dem_transform = src.transform
    dem_crs = src.crs


# ------------------------------------------------------------
# Resample DEM onto NDVI grid
# ------------------------------------------------------------

dem_resampled = np.full(
    ndvi_shape,
    np.nan,
    dtype="float32"
)


reproject(
    source=dem,
    destination=dem_resampled,

    src_transform=dem_transform,
    src_crs=dem_crs,

    dst_transform=ndvi_transform,
    dst_crs=ndvi_crs,

    resampling=Resampling.bilinear
)


# ------------------------------------------------------------
# Common valid pixels
# ------------------------------------------------------------

valid = (
    np.isfinite(ndvi)
    &
    np.isfinite(dem_resampled)
)


elevation = dem_resampled[valid]
ndvi_values = ndvi[valid]


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("=" * 60)
print("ELEVATION - NDVI ANALYSIS")
print("=" * 60)

print(f"Valid pixels: {len(elevation):,}")

print()
print("Elevation:")
print(f"Minimum: {np.min(elevation):.1f} m")
print(f"Maximum: {np.max(elevation):.1f} m")
print(f"Mean:    {np.mean(elevation):.1f} m")
print(f"Median:  {np.median(elevation):.1f} m")

print()
print("NDVI:")
print(f"Minimum: {np.min(ndvi_values):.3f}")
print(f"Maximum: {np.max(ndvi_values):.3f}")
print(f"Mean:    {np.mean(ndvi_values):.3f}")
print(f"Median:  {np.median(ndvi_values):.3f}")


# ------------------------------------------------------------
# Elevation bins
# ------------------------------------------------------------

bins = np.arange(
    np.floor(np.min(elevation) / 250) * 250,
    np.ceil(np.max(elevation) / 250) * 250 + 250,
    250
)

bin_centers = []
mean_ndvi = []
std_ndvi = []
pixel_counts = []


for lower, upper in zip(bins[:-1], bins[1:]):

    mask = (
        (elevation >= lower)
        &
        (elevation < upper)
    )

    if np.sum(mask) < 100:
        continue

    bin_centers.append((lower + upper) / 2)

    mean_ndvi.append(
        np.mean(ndvi_values[mask])
    )

    std_ndvi.append(
        np.std(ndvi_values[mask])
    )

    pixel_counts.append(
        np.sum(mask)
    )


# ------------------------------------------------------------
# Print elevation-bin results
# ------------------------------------------------------------

print()
print("Elevation-bin NDVI:")
print("-" * 60)

for elev, mean, std, count in zip(
    bin_centers,
    mean_ndvi,
    std_ndvi,
    pixel_counts
):

    print(
        f"{elev:7.0f} m | "
        f"Mean NDVI = {mean:.3f} | "
        f"SD = {std:.3f} | "
        f"Pixels = {count:,}"
    )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.errorbar(
    bin_centers,
    mean_ndvi,
    yerr=std_ndvi,
    fmt="o-",
    capsize=4
)

plt.xlabel("Elevation (m)")
plt.ylabel("Mean NDVI")
plt.title(
    "Relationship Between Elevation and NDVI\n"
    "Gokyo Valley, 2025"
)

plt.grid(alpha=0.25)

plt.tight_layout()


figure_path = (
    FIGURE_DIR /
    "Elevation_NDVI_2025.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("Figure saved to:")
print(figure_path)

print()
print("Terrain analysis completed.")