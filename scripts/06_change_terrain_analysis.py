from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read NDVI 2017
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2017.tif") as src:

    ndvi_2017 = src.read(1).astype("float32")

    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_shape = src.shape


# ============================================================
# Read NDVI 2025
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2025.tif") as src:

    ndvi_2025 = src.read(1).astype("float32")


# ============================================================
# Validate dimensions
# ============================================================

if ndvi_2017.shape != ndvi_2025.shape:

    raise ValueError(
        f"NDVI dimensions do not match: "
        f"{ndvi_2017.shape} vs {ndvi_2025.shape}"
    )


# ============================================================
# Clean NDVI
# ============================================================

ndvi_2017[
    (ndvi_2017 < -1) |
    (ndvi_2017 > 1)
] = np.nan

ndvi_2025[
    (ndvi_2025 < -1) |
    (ndvi_2025 > 1)
] = np.nan


# ============================================================
# NDVI change
# ============================================================

ndvi_change = (
    ndvi_2025 -
    ndvi_2017
)


# ============================================================
# Read DEM
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_DEM.tif") as src:

    dem = src.read(1).astype("float32")

    dem_transform = src.transform
    dem_crs = src.crs


# ============================================================
# Resample DEM to NDVI grid
# ============================================================

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


# ============================================================
# Calculate slope
# ============================================================

pixel_x = abs(ndvi_transform.a)
pixel_y = abs(ndvi_transform.e)

latitude = 27.97

meters_per_degree_lat = 111320
meters_per_degree_lon = (
    111320 *
    np.cos(np.deg2rad(latitude))
)

dx = pixel_x * meters_per_degree_lon
dy = pixel_y * meters_per_degree_lat

dz_dy, dz_dx = np.gradient(
    dem_resampled,
    dy,
    dx
)

slope = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx ** 2 +
            dz_dy ** 2
        )
    )
)


# ============================================================
# Common valid pixels
# ============================================================

valid = (
    np.isfinite(ndvi_change)
    &
    np.isfinite(dem_resampled)
    &
    np.isfinite(slope)
)

change_values = ndvi_change[valid]
elevation_values = dem_resampled[valid]
slope_values = slope[valid]


print()
print("=" * 65)
print("NDVI CHANGE VS TERRAIN")
print("=" * 65)

print(f"Common valid pixels: {len(change_values):,}")

print()
print("NDVI change:")
print(f"Mean:   {np.mean(change_values):.4f}")
print(f"Median: {np.median(change_values):.4f}")

print()
print("Elevation:")
print(f"Minimum: {np.min(elevation_values):.1f} m")
print(f"Maximum: {np.max(elevation_values):.1f} m")


# ============================================================
# NDVI change by elevation
# ============================================================

elevation_bins = np.arange(
    4400,
    6201,
    200
)

elevation_centers = []
elevation_change = []
elevation_counts = []


print()
print("NDVI change by elevation:")
print("-" * 65)


for lower, upper in zip(
    elevation_bins[:-1],
    elevation_bins[1:]
):

    mask = (
        (elevation_values >= lower)
        &
        (elevation_values < upper)
    )

    count = np.sum(mask)

    if count < 100:
        continue

    mean_change = np.mean(
        change_values[mask]
    )

    center = (lower + upper) / 2

    elevation_centers.append(center)
    elevation_change.append(mean_change)
    elevation_counts.append(count)

    print(
        f"{lower:4.0f}–{upper:4.0f} m | "
        f"Mean change = {mean_change:+.4f} | "
        f"Pixels = {count:,}"
    )


# ============================================================
# Plot elevation vs NDVI change
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    elevation_centers,
    elevation_change,
    "o-"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Elevation (m)")
plt.ylabel("Mean NDVI change (2025 − 2017)")

plt.title(
    "NDVI Change Along the Elevation Gradient\n"
    "Gokyo Valley, 2017–2025"
)

plt.grid(alpha=0.25)

plt.tight_layout()

figure_path = (
    FIGURE_DIR /
    "NDVI_Change_Elevation_2017_2025.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# NDVI change by slope
# ============================================================

slope_bins = [
    0,
    10,
    20,
    30,
    40,
    50,
    90
]

print()
print("NDVI change by slope:")
print("-" * 65)


for lower, upper in zip(
    slope_bins[:-1],
    slope_bins[1:]
):

    mask = (
        (slope_values >= lower)
        &
        (slope_values < upper)
    )

    count = np.sum(mask)

    if count < 100:
        continue

    mean_change = np.mean(
        change_values[mask]
    )

    print(
        f"{lower:2.0f}–{upper:2.0f}° | "
        f"Mean change = {mean_change:+.4f} | "
        f"Pixels = {count:,}"
    )


print()
print("Figure saved to:")
print(figure_path)

print()
print("NDVI change-terrain analysis completed.")