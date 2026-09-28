import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

ndvi_2017_path = "data/processed/Gokyo_NDVI_2017.tif"
ndvi_2025_path = "data/processed/Gokyo_NDVI_2025.tif"
dem_path = "data/processed/Gokyo_DEM.tif"

output_path = "outputs/tables/spatial_cross_validation.csv"


# ---------------------------------------------------------
# Read NDVI
# ---------------------------------------------------------

with rasterio.open(ndvi_2017_path) as src:
    ndvi_2017 = src.read(1).astype("float32")
    ndvi_transform = src.transform
    ndvi_profile = src.profile

with rasterio.open(ndvi_2025_path) as src:
    ndvi_2025 = src.read(1).astype("float32")


# ---------------------------------------------------------
# Read and resample DEM
# ---------------------------------------------------------

with rasterio.open(dem_path) as src:

    dem = np.empty_like(ndvi_2025, dtype="float32")

    reproject(
        source=src.read(1),
        destination=dem,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ndvi_transform,
        dst_crs=ndvi_profile["crs"],
        resampling=Resampling.bilinear
    )


# ---------------------------------------------------------
# NDVI change
# ---------------------------------------------------------

ndvi_change = ndvi_2025 - ndvi_2017

valid = (
    np.isfinite(ndvi_2017)
    & np.isfinite(ndvi_2025)
    & np.isfinite(dem)
    & (ndvi_2017 >= -1)
    & (ndvi_2017 <= 1)
    & (ndvi_2025 >= -1)
    & (ndvi_2025 <= 1)
)


# ---------------------------------------------------------
# Slope
# ---------------------------------------------------------

pixel_size_x = abs(ndvi_transform.a)
pixel_size_y = abs(ndvi_transform.e)

mean_latitude = 27.97

meters_per_degree_lat = 111320
meters_per_degree_lon = (
    111320 * np.cos(np.deg2rad(mean_latitude))
)

dx = pixel_size_x * meters_per_degree_lon
dy = pixel_size_y * meters_per_degree_lat

dz_dy, dz_dx = np.gradient(
    dem,
    dy,
    dx
)

slope = np.degrees(
    np.arctan(
        np.sqrt(dz_dx**2 + dz_dy**2)
    )
)


# ---------------------------------------------------------
# Aspect
# ---------------------------------------------------------

aspect = (
    np.degrees(
        np.arctan2(-dz_dx, dz_dy)
    )
    + 360
) % 360

northness = np.cos(
    np.deg2rad(aspect)
)

eastness = np.sin(
    np.deg2rad(aspect)
)


# ---------------------------------------------------------
# Spatial blocks
# ---------------------------------------------------------

rows, cols = np.indices(
    ndvi_2025.shape
)

block_size = 150

block_row = rows // block_size
block_col = cols // block_size

block_id = (
    block_row * (cols.max() // block_size + 1)
    + block_col
)


# ---------------------------------------------------------
# Build modelling dataset
# ---------------------------------------------------------

X = np.column_stack([
    dem[valid],
    slope[valid],
    northness[valid],
    eastness[valid],
    ndvi_2017[valid]
])

y = ndvi_change[valid]

groups = block_id[valid]


# ---------------------------------------------------------
# Reproducible sample
# ---------------------------------------------------------

rng = np.random.default_rng(42)

sample_size = min(
    100000,
    len(y)
)

sample_idx = rng.choice(
    len(y),
    size=sample_size,
    replace=False
)

X = X[sample_idx]
y = y[sample_idx]
groups = groups[sample_idx]


print("Total valid pixels:", len(y))
print(
    "Spatial blocks:",
    len(np.unique(groups))
)


# ---------------------------------------------------------
# 5-fold spatial cross-validation
# ---------------------------------------------------------

gkf = GroupKFold(
    n_splits=5
)

fold_results = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups=groups),
    start=1
):

    print(f"\nFold {fold}")
    print("-" * 20)

    print(
        "Training pixels:",
        len(train_idx)
    )

    print(
        "Testing pixels:",
        len(test_idx)
    )

    print(
        "Training blocks:",
        len(np.unique(groups[train_idx]))
    )

    print(
        "Testing blocks:",
        len(np.unique(groups[test_idx]))
    )


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X[train_idx],
        y[train_idx]
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_pred = model.predict(
        X[test_idx]
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    r2 = r2_score(
        y[test_idx],
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y[test_idx],
            y_pred
        )
    )


    print(
        f"R²   : {r2:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )


    fold_results.append({
        "fold": fold,
        "r2": r2,
        "rmse": rmse,
        "train_pixels": len(train_idx),
        "test_pixels": len(test_idx),
        "train_blocks": len(
            np.unique(groups[train_idx])
        ),
        "test_blocks": len(
            np.unique(groups[test_idx])
        )
    })


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

results = pd.DataFrame(
    fold_results
)

mean_r2 = results["r2"].mean()
std_r2 = results["r2"].std()

mean_rmse = results["rmse"].mean()
std_rmse = results["rmse"].std()


print("\n==============================")
print("5-FOLD SPATIAL VALIDATION")
print("==============================")

print(
    f"Mean R²   : {mean_r2:.4f}"
)

print(
    f"SD R²     : {std_r2:.4f}"
)

print(
    f"Mean RMSE : {mean_rmse:.4f}"
)

print(
    f"SD RMSE   : {std_rmse:.4f}"
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

results.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)