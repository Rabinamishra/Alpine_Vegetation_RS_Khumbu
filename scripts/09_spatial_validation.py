import numpy as np
import pandas as pd
import rasterio

from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

ndvi_2017_path = "data/processed/Gokyo_NDVI_2017.tif"
ndvi_2025_path = "data/processed/Gokyo_NDVI_2025.tif"
dem_path = "data/processed/Gokyo_DEM.tif"

output_path = "outputs/tables/spatial_validation_results.csv"


# ---------------------------------------------------------
# Read NDVI rasters
# ---------------------------------------------------------

with rasterio.open(ndvi_2017_path) as src:
    ndvi_2017 = src.read(1).astype("float32")
    ndvi_profile = src.profile
    ndvi_transform = src.transform

with rasterio.open(ndvi_2025_path) as src:
    ndvi_2025 = src.read(1).astype("float32")


# ---------------------------------------------------------
# Read and resample DEM to NDVI grid
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
# Calculate NDVI change
# ---------------------------------------------------------

valid = (
    np.isfinite(ndvi_2017)
    & np.isfinite(ndvi_2025)
    & np.isfinite(dem)
    & (ndvi_2017 >= -1)
    & (ndvi_2017 <= 1)
    & (ndvi_2025 >= -1)
    & (ndvi_2025 <= 1)
)

ndvi_change = ndvi_2025 - ndvi_2017


# ---------------------------------------------------------
# Calculate slope
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

dz_dy, dz_dx = np.gradient(dem, dy, dx)

slope = np.degrees(
    np.arctan(
        np.sqrt(dz_dx**2 + dz_dy**2)
    )
)


# ---------------------------------------------------------
# Calculate aspect
# ---------------------------------------------------------

aspect = (
    np.degrees(
        np.arctan2(-dz_dx, dz_dy)
    )
    + 360
) % 360

northness = np.cos(np.deg2rad(aspect))
eastness = np.sin(np.deg2rad(aspect))


# ---------------------------------------------------------
# Create spatial blocks
# ---------------------------------------------------------

rows, cols = np.indices(ndvi_2025.shape)

# Approximately 150 x 150 pixel blocks
# At 10 m resolution this is roughly 1.5 km blocks.
block_size = 150

block_row = rows // block_size
block_col = cols // block_size

block_id = (
    block_row * (cols.max() // block_size + 1)
    + block_col
)


# ---------------------------------------------------------
# Extract valid pixels
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

print("Total valid pixels:", len(y))
print("Number of spatial blocks:", len(np.unique(groups)))


# ---------------------------------------------------------
# Reproducible sample
# ---------------------------------------------------------

rng = np.random.default_rng(42)

sample_size = min(100000, len(y))

sample_idx = rng.choice(
    len(y),
    size=sample_size,
    replace=False
)

X = X[sample_idx]
y = y[sample_idx]
groups = groups[sample_idx]

print("Model sample:", len(y))


# ---------------------------------------------------------
# Spatial block train/test split
# ---------------------------------------------------------

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

print("Training pixels:", len(train_idx))
print("Testing pixels:", len(test_idx))

print(
    "Training blocks:",
    len(np.unique(groups[train_idx]))
)

print(
    "Testing blocks:",
    len(np.unique(groups[test_idx]))
)


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)


print("\nSpatial validation results")
print("--------------------------")

print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

feature_names = [
    "elevation",
    "slope",
    "northness",
    "eastness",
    "ndvi_2017"
]

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values(
    "importance",
    ascending=False
)

print("\nFeature importance:")
print(importance_df)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = pd.DataFrame({
    "validation_type": [
        "spatial_block"
    ],
    "sample_size": [
        sample_size
    ],
    "train_pixels": [
        len(train_idx)
    ],
    "test_pixels": [
        len(test_idx)
    ],
    "train_blocks": [
        len(np.unique(groups[train_idx]))
    ],
    "test_blocks": [
        len(np.unique(groups[test_idx]))
    ],
    "r2": [
        r2
    ],
    "rmse": [
        rmse
    ]
})

results.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved results to: {output_path}"
)