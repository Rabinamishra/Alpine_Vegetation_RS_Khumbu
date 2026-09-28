from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# UiB Alpine Vegetation Project
# NDVI Change Analysis: 2017–2025
# Gokyo Valley, Khumbu Himal, Nepal
# ============================================================


# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Read 2017 NDVI
# ------------------------------------------------------------

path_2017 = DATA_DIR / "Gokyo_NDVI_2017.tif"

with rasterio.open(path_2017) as src:

    ndvi_2017 = src.read(1).astype("float32")

    profile = src.profile
    transform = src.transform
    crs = src.crs


# ------------------------------------------------------------
# 3. Read 2025 NDVI
# ------------------------------------------------------------

path_2025 = DATA_DIR / "Gokyo_NDVI_2025.tif"

with rasterio.open(path_2025) as src:

    ndvi_2025 = src.read(1).astype("float32")


# ------------------------------------------------------------
# 4. Check raster dimensions
# ------------------------------------------------------------

if ndvi_2017.shape != ndvi_2025.shape:

    raise ValueError(
        f"Raster dimensions do not match: "
        f"2017={ndvi_2017.shape}, "
        f"2025={ndvi_2025.shape}"
    )


# ------------------------------------------------------------
# 5. Mask invalid NDVI values
# ------------------------------------------------------------

ndvi_2017[
    (ndvi_2017 < -1) |
    (ndvi_2017 > 1)
] = np.nan

ndvi_2025[
    (ndvi_2025 < -1) |
    (ndvi_2025 > 1)
] = np.nan


# ------------------------------------------------------------
# 6. Create common-valid-pixel mask
# ------------------------------------------------------------
#
# Only compare pixels that contain valid NDVI
# observations in BOTH 2017 and 2025.
#

common_valid = (
    np.isfinite(ndvi_2017)
    &
    np.isfinite(ndvi_2025)
)


# ------------------------------------------------------------
# 7. Calculate NDVI change
# ------------------------------------------------------------

ndvi_change = np.full_like(
    ndvi_2017,
    np.nan,
    dtype="float32"
)

ndvi_change[common_valid] = (
    ndvi_2025[common_valid]
    -
    ndvi_2017[common_valid]
)


# ------------------------------------------------------------
# 8. Extract valid change values
# ------------------------------------------------------------

valid_change = ndvi_change[
    np.isfinite(ndvi_change)
]


# ------------------------------------------------------------
# 9. Summary statistics
# ------------------------------------------------------------

print()
print("=" * 60)
print("NDVI CHANGE: 2025 - 2017")
print("=" * 60)

print(
    f"Common valid pixels: "
    f"{len(valid_change):,}"
)

print(
    f"Mean change: "
    f"{np.mean(valid_change):.4f}"
)

print(
    f"Median change: "
    f"{np.median(valid_change):.4f}"
)

print(
    f"Standard deviation: "
    f"{np.std(valid_change):.4f}"
)

print(
    f"Minimum change: "
    f"{np.min(valid_change):.4f}"
)

print(
    f"Maximum change: "
    f"{np.max(valid_change):.4f}"
)


# ------------------------------------------------------------
# 10. Percentiles
# ------------------------------------------------------------

p05 = np.percentile(valid_change, 5)
p25 = np.percentile(valid_change, 25)
p75 = np.percentile(valid_change, 75)
p95 = np.percentile(valid_change, 95)

print()
print("Change percentiles:")
print(
    f"5th percentile:  "
    f"{p05:.4f}"
)
print(
    f"25th percentile: "
    f"{p25:.4f}"
)
print(
    f"75th percentile: "
    f"{p75:.4f}"
)
print(
    f"95th percentile: "
    f"{p95:.4f}"
)


# ------------------------------------------------------------
# 11. Exploratory change categories
# ------------------------------------------------------------
#
# These are simple visualization thresholds.
# They are NOT species-specific vegetation-change thresholds.
#

increase = (
    np.sum(valid_change > 0.10)
    /
    len(valid_change)
    *
    100
)

stable = (
    np.sum(
        (valid_change >= -0.10)
        &
        (valid_change <= 0.10)
    )
    /
    len(valid_change)
    *
    100
)

decrease = (
    np.sum(valid_change < -0.10)
    /
    len(valid_change)
    *
    100
)


print()
print("Exploratory change categories:")
print(
    f"Increase > +0.10: "
    f"{increase:.2f}%"
)

print(
    f"Stable (-0.10 to +0.10): "
    f"{stable:.2f}%"
)

print(
    f"Decrease < -0.10: "
    f"{decrease:.2f}%"
)


# ------------------------------------------------------------
# 12. Create NDVI change map
# ------------------------------------------------------------

plt.figure(figsize=(9, 7))

image = plt.imshow(
    ndvi_change,
    cmap="RdYlGn",
    vmin=-0.5,
    vmax=0.5
)

plt.colorbar(
    image,
    label="NDVI change (2025 − 2017)"
)

plt.title(
    "NDVI Change in the Gokyo Valley\n"
    "2017–2025"
)

plt.axis("off")

plt.tight_layout()


# ------------------------------------------------------------
# 13. Save figure
# ------------------------------------------------------------

figure_path = (
    FIGURE_DIR /
    "NDVI_change_2017_2025.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Final message
# ------------------------------------------------------------

print()
print("Figure saved to:")
print(figure_path)
print()
print("NDVI change analysis completed.")