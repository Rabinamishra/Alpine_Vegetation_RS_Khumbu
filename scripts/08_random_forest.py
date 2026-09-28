from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance


# ============================================================
# Paths
# ============================================================

PROJECT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT / "data" / "processed"
FIGURE_DIR = PROJECT / "outputs" / "figures"
TABLE_DIR = PROJECT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Read NDVI 2017
# ============================================================

with rasterio.open(DATA_DIR / "Gokyo_NDVI_2017.tif") as src:

    ndvi_2017 = src.read(1).astype("float32")

    transform = src.transform
    crs = src.crs
    shape = src.shape


# ============================================================
# Read NDVI 2025
# ============================================================

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
# Target variable
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
# Calculate slope
# ============================================================

pixel_x = abs(transform.a)
pixel_y = abs(transform.e)

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
# Convert aspect to continuous variables
# ============================================================

northness = np.cos(
    np.deg2rad(aspect)
)

eastness = np.sin(
    np.deg2rad(aspect)
)


# ============================================================
# Build valid-pixel mask
# ============================================================

valid = (
    np.isfinite(ndvi_2017)
    &
    np.isfinite(ndvi_change)
    &
    np.isfinite(dem_resampled)
    &
    np.isfinite(slope)
    &
    np.isfinite(northness)
    &
    np.isfinite(eastness)
)


# ============================================================
# Extract values
# ============================================================

elevation = dem_resampled[valid]

slope_values = slope[valid]

northness_values = northness[valid]

eastness_values = eastness[valid]

ndvi_2017_values = ndvi_2017[valid]

change_values = ndvi_change[valid]


print()
print("=" * 65)
print("RANDOM FOREST MODEL")
print("=" * 65)

print(
    f"Available valid pixels: "
    f"{len(change_values):,}"
)


# ============================================================
# Create modelling DataFrame
# ============================================================

df = pd.DataFrame({
    "elevation": elevation,
    "slope": slope_values,
    "northness": northness_values,
    "eastness": eastness_values,
    "ndvi_2017": ndvi_2017_values,
    "ndvi_change": change_values
})


# ============================================================
# Remove extreme target values
# ============================================================

df = df[
    (df["ndvi_change"] >= -1) &
    (df["ndvi_change"] <= 1)
].copy()


print(
    f"Valid modelling pixels: "
    f"{len(df):,}"
)


# ============================================================
# Sample manageable dataset
# ============================================================

MAX_SAMPLES = 100_000

if len(df) > MAX_SAMPLES:

    df = df.sample(
        n=MAX_SAMPLES,
        random_state=42
    )


print(
    f"Model sample size: "
    f"{len(df):,}"
)


# ============================================================
# Predictor and target
# ============================================================

features = [
    "elevation",
    "slope",
    "northness",
    "eastness",
    "ndvi_2017"
]

X = df[features]

y = df["ndvi_change"]


# ============================================================
# Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,

    test_size=0.20,

    random_state=42
)


print()
print(
    f"Training samples: "
    f"{len(X_train):,}"
)

print(
    f"Testing samples: "
    f"{len(X_test):,}"
)


# ============================================================
# Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# Predictions
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# Model performance
# ============================================================

r2 = r2_score(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)


print()
print("MODEL PERFORMANCE")
print("-" * 65)

print(
    f"R²:   {r2:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)


# ============================================================
# Built-in feature importance
# ============================================================

importance = model.feature_importances_


importance_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(
    "importance",
    ascending=False
)


print()
print("RANDOM FOREST FEATURE IMPORTANCE")
print("-" * 65)

for _, row in importance_df.iterrows():

    print(
        f"{row['feature']:>12}: "
        f"{row['importance']:.4f}"
    )


# ============================================================
# Permutation importance
# ============================================================

perm = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

perm_df = pd.DataFrame({
    "feature": features,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(
    "importance_mean",
    ascending=False
)


print()
print("PERMUTATION IMPORTANCE")
print("-" * 65)

for _, row in perm_df.iterrows():

    print(
        f"{row['feature']:>12}: "
        f"{row['importance_mean']:.4f} "
        f"+/- {row['importance_std']:.4f}"
    )


# ============================================================
# Save feature importance
# ============================================================

importance_df.to_csv(
    TABLE_DIR /
    "random_forest_feature_importance.csv",
    index=False
)

perm_df.to_csv(
    TABLE_DIR /
    "random_forest_permutation_importance.csv",
    index=False
)


# ============================================================
# Feature importance plot
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Random Forest feature importance"
)

plt.ylabel("Predictor")

plt.title(
    "Predictor Importance for NDVI Change\n"
    "Gokyo Valley, 2017–2025"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

importance_figure = (
    FIGURE_DIR /
    "Random_Forest_Feature_Importance.png"
)

plt.savefig(
    importance_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Observed vs predicted
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    s=8,
    alpha=0.25
)

min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "Observed NDVI change"
)

plt.ylabel(
    "Predicted NDVI change"
)

plt.title(
    "Observed vs Predicted NDVI Change\n"
    f"Random Forest (R² = {r2:.3f})"
)

plt.grid(alpha=0.25)

plt.tight_layout()

prediction_figure = (
    FIGURE_DIR /
    "Random_Forest_Observed_vs_Predicted.png"
)

plt.savefig(
    prediction_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Finish
# ============================================================

print()
print("Figures saved:")
print(importance_figure)
print(prediction_figure)

print()
print("Tables saved:")
print(
    TABLE_DIR /
    "random_forest_feature_importance.csv"
)

print(
    TABLE_DIR /
    "random_forest_permutation_importance.csv"
)

print()
print("Random Forest analysis completed.")