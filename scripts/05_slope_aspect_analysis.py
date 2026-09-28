from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read NDVI 2025
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2025.tif") as src:

    ndvi = src.read(1).astype("float32")

    ndvi_transform = src.transform
    ndvi_crs = src.crs
    ndvi_shape = src.shape


ndvi[(ndvi < -1) | (ndvi > 1)] = np.nan


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
# Calculate terrain derivatives
# ============================================================

# Pixel dimensions in degrees
pixel_x = abs(ndvi_transform.a)
pixel_y = abs(ndvi_transform.e)

# Convert approximate degree spacing to metres
lat = 27.97

meters_per_degree_lat = 111320
meters_per_degree_lon = (
    111320 * np.cos(np.deg2rad(lat))
)

dx = pixel_x * meters_per_degree_lon
dy = pixel_y * meters_per_degree_lat


# Gradient
dz_dy, dz_dx = np.gradient(
    dem_resampled,
    dy,
    dx
)


# Slope in degrees
slope = np.degrees(
    np.arctan(
        np.sqrt(
            dz_dx ** 2 +
            dz_dy ** 2
        )
    )
)


# Aspect
aspect = np.degrees(
    np.arctan2(
        dz_dx,
        -dz_dy
    )
)

aspect = np.where(
    aspect < 0,
    360 + aspect,
    aspect
)


# ============================================================
# Common valid pixels
# ============================================================

valid = (
    np.isfinite(ndvi)
    &
    np.isfinite(dem_resampled)
    &
    np.isfinite(slope)
    &
    np.isfinite(aspect)
)

ndvi_values = ndvi[valid]
slope_values = slope[valid]
aspect_values = aspect[valid]
elevation_values = dem_resampled[valid]


# ============================================================
# Basic statistics
# ============================================================

print()
print("=" * 60)
print("SLOPE + ASPECT ANALYSIS")
print("=" * 60)

print(f"Valid pixels: {len(ndvi_values):,}")

print()
print("Slope:")
print(f"Minimum: {np.min(slope_values):.2f} degrees")
print(f"Maximum: {np.max(slope_values):.2f} degrees")
print(f"Mean:    {np.mean(slope_values):.2f} degrees")
print(f"Median:  {np.median(slope_values):.2f} degrees")

print()
print("Aspect:")
print(f"Minimum: {np.min(aspect_values):.2f} degrees")
print(f"Maximum: {np.max(aspect_values):.2f} degrees")


# ============================================================
# Slope bins
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

slope_labels = [
    "0–10",
    "10–20",
    "20–30",
    "30–40",
    "40–50",
    "50+"
]

print()
print("NDVI by slope:")
print("-" * 60)

slope_means = []
slope_centers = []

for i in range(len(slope_bins) - 1):

    lower = slope_bins[i]
    upper = slope_bins[i + 1]

    mask = (
        (slope_values >= lower)
        &
        (slope_values < upper)
    )

    count = np.sum(mask)

    if count == 0:
        continue

    mean_ndvi = np.mean(ndvi_values[mask])

    print(
        f"{slope_labels[i]:>6}° | "
        f"Mean NDVI = {mean_ndvi:.3f} | "
        f"Pixels = {count:,}"
    )

    slope_means.append(mean_ndvi)
    slope_centers.append((lower + upper) / 2)


# ============================================================
# Plot NDVI vs slope
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    slope_centers,
    slope_means,
    "o-"
)

plt.xlabel("Slope (degrees)")
plt.ylabel("Mean NDVI")
plt.title(
    "Relationship Between Slope and NDVI\n"
    "Gokyo Valley, 2025"
)

plt.grid(alpha=0.25)

plt.tight_layout()

slope_figure = (
    FIGURE_DIR /
    "Slope_NDVI_2025.png"
)

plt.savefig(
    slope_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Aspect sectors
# ============================================================

aspect_sectors = {
    "North": (337.5, 360),
    "North_2": (0, 22.5),
    "Northeast": (22.5, 67.5),
    "East": (67.5, 112.5),
    "Southeast": (112.5, 157.5),
    "South": (157.5, 202.5),
    "Southwest": (202.5, 247.5),
    "West": (247.5, 292.5),
    "Northwest": (292.5, 337.5)
}


print()
print("NDVI by aspect:")
print("-" * 60)


aspect_names = []
aspect_means = []
aspect_counts = []


for name, (lower, upper) in aspect_sectors.items():

    if name == "North":
        mask = (
            (aspect_values >= lower)
            |
            (aspect_values < 22.5)
        )
    elif name == "North_2":
        continue
    else:
        mask = (
            (aspect_values >= lower)
            &
            (aspect_values < upper)
        )

    count = np.sum(mask)

    if count == 0:
        continue

    mean_ndvi = np.mean(ndvi_values[mask])

    display_name = name.replace("_", " ")

    print(
        f"{display_name:>10} | "
        f"Mean NDVI = {mean_ndvi:.3f} | "
        f"Pixels = {count:,}"
    )

    aspect_names.append(display_name)
    aspect_means.append(mean_ndvi)
    aspect_counts.append(count)


# ============================================================
# Plot aspect relationship
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    aspect_names,
    aspect_means
)

plt.xlabel("Aspect direction")
plt.ylabel("Mean NDVI")
plt.title(
    "Relationship Between Aspect and NDVI\n"
    "Gokyo Valley, 2025"
)

plt.xticks(rotation=30)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

aspect_figure = (
    FIGURE_DIR /
    "Aspect_NDVI_2025.png"
)

plt.savefig(
    aspect_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Finish
# ============================================================

print()
print("Figures saved:")
print(slope_figure)
print(aspect_figure)

print()
print("Slope + aspect analysis completed.")