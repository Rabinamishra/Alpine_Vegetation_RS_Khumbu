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
# Read NDVI
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2017.tif") as src:
    ndvi_2017 = src.read(1).astype("float32")
    transform = src.transform
    crs = src.crs
    shape = src.shape

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2025.tif") as src:
    ndvi_2025 = src.read(1).astype("float32")


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

ndvi_change = ndvi_2025 - ndvi_2017


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
    shape,
    np.nan,
    dtype="float32"
)

reproject(
    source=dem,
    destination=dem_resampled,

    src_transform=dem_transform,
    src_crs=dem_crs,

    dst_transform=transform,
    dst_crs=crs,

    resampling=Resampling.bilinear
)


# ============================================================
# Calculate slope derivatives
# ============================================================

pixel_x = abs(transform.a)
pixel_y = abs(transform.e)

latitude = 27.97

meters_per_degree_lat = 111320
meters_per_degree_lon = (
    111320 * np.cos(np.deg2rad(latitude))
)

dx = pixel_x * meters_per_degree_lon
dy = pixel_y * meters_per_degree_lat

dz_dy, dz_dx = np.gradient(
    dem_resampled,
    dy,
    dx
)


# ============================================================
# Calculate aspect
# ============================================================

aspect = np.degrees(
    np.arctan2(
        dz_dx,
        -dz_dy
    )
)

aspect = np.where(
    aspect < 0,
    aspect + 360,
    aspect
)


# ============================================================
# Common valid pixels
# ============================================================

valid = (
    np.isfinite(ndvi_change)
    &
    np.isfinite(aspect)
)

change_values = ndvi_change[valid]
aspect_values = aspect[valid]


# ============================================================
# Aspect sectors
# ============================================================

sectors = [
    ("North", 337.5, 22.5),
    ("Northeast", 22.5, 67.5),
    ("East", 67.5, 112.5),
    ("Southeast", 112.5, 157.5),
    ("South", 157.5, 202.5),
    ("Southwest", 202.5, 247.5),
    ("West", 247.5, 292.5),
    ("Northwest", 292.5, 337.5),
]


print()
print("=" * 65)
print("NDVI CHANGE VS ASPECT")
print("=" * 65)

print(f"Common valid pixels: {len(change_values):,}")

print()
print("Mean NDVI change by aspect:")
print("-" * 65)


names = []
means = []
counts = []


for name, lower, upper in sectors:

    if name == "North":

        mask = (
            (aspect_values >= lower)
            |
            (aspect_values < upper)
        )

    else:

        mask = (
            (aspect_values >= lower)
            &
            (aspect_values < upper)
        )

    count = np.sum(mask)

    if count < 100:
        continue

    mean_change = np.mean(
        change_values[mask]
    )

    names.append(name)
    means.append(mean_change)
    counts.append(count)

    print(
        f"{name:>10} | "
        f"Mean change = {mean_change:+.4f} | "
        f"Pixels = {count:,}"
    )


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    names,
    means
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Aspect direction")
plt.ylabel("Mean NDVI change (2025 − 2017)")

plt.title(
    "NDVI Change by Aspect\n"
    "Gokyo Valley, 2017–2025"
)

plt.xticks(rotation=30)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()


figure_path = (
    FIGURE_DIR /
    "NDVI_Change_Aspect_2017_2025.png"
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
print("Aspect-change analysis completed.")