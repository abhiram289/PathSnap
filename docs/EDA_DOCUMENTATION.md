# PathSnap – Exploratory Data Analysis Documentation

## 1. Introduction

Exploratory Data Analysis (EDA) is performed to understand the characteristics of GPS trajectories and road network data before applying machine learning.

The EDA implementation is located in:

`src/eda/explore.py`

The analysis generates summary CSV files, diagnostic figures and interactive Folium maps.

## 2. Objectives of EDA

- Understand GPS trajectory characteristics.
- Analyse GPS sampling intervals.
- Examine trip lengths and route sizes.
- Identify missing or invalid values.
- Understand geographic coordinate distributions.
- Visualize GPS trajectories and road networks.
- Generate dataset inventory reports.

## 3. Dataset Inventory

The committed dataset inventory reports:

| Metric | Training | Student Testing |
|---|---:|---:|
| Number of trips | 80 | 10 |
| Total GPS points | 208,417 | 19,160 |
| Average GPS points per trip | 2,605.21 | 1,916 |
| Total traversed arcs | 31,336 | 3,477 |
| Missing values | 0 | 0 |
| Invalid coordinates | 0 | 0 |

## 4. Data Quality Analysis

The EDA script checks:

### Missing Values

Checks whether longitude, latitude or time values are missing.

### Invalid Coordinates

Checks whether longitude lies outside [-180, 180] or latitude lies outside [-90, 90].

The committed inventory reports zero missing values and zero invalid coordinates in both partitions.

## 5. Visualizations

### 5.1 Distribution of GPS Observations and Route Arcs

File:

`results/figures/eda_trip_lengths_and_points.png`

This visualization contains two histograms:

- Distribution of GPS observations per training trip.
- Distribution of ground-truth route arcs per training trip.

**What it tells us:**

It helps identify whether trips contain similar numbers of GPS observations or whether there is significant variation in trip lengths.

It also shows how route complexity varies across trips.

### 5.2 GPS Sampling Interval Analysis

File:

`results/figures/eda_sampling_intervals.png`

This visualization contains:

- GPS latitude progression over time.
- Frequency distribution of GPS sampling intervals.

**What it tells us:**

The sampling interval represents the time difference between two consecutive GPS observations.

Regular intervals indicate consistent GPS sampling.

Larger intervals may indicate sensor delays or missing observations.

The README reports a median sampling interval of 1 second and 98.4% of consecutive observations within the 0.8–1.2 second interval.

### 5.3 Interactive GPS Maps

Generated files:

- `results/maps/interactive_map_trip_00000001.html`
- `results/maps/interactive_map_trip_00000000_test.html`

The Folium maps display GPS trajectories and road network information.

The visualizations help compare the recorded GPS path with the underlying road network and ground-truth route where available.

## 6. Spatial Coordinate Analysis

GPS coordinates are originally represented using latitude and longitude in geographic degrees.

However, calculating Euclidean distances directly in degrees is not appropriate for accurate local distance calculations.

The preprocessing pipeline converts coordinates into local metric coordinates using an equirectangular approximation.

This allows distances to be represented in metres.

## 7. Road Network Exploration

The road network is represented through:

- Nodes: Geographic locations.
- Arcs: Connections between nodes.
- Routes: Ordered sequences of road arcs.

Understanding this structure is important for generating candidate road segments and reconstructing continuous routes.

## 8. Generated EDA Outputs

| Output | Purpose |
|---|---|
| dataset_inventory.csv | Overall dataset summary |
| trip_summary_train.csv | Training trip statistics |
| trip_summary_test.csv | Testing trip statistics |
| eda_trip_lengths_and_points.png | Trip and route distributions |
| eda_sampling_intervals.png | GPS sampling analysis |
| Interactive Folium maps | Spatial visualization |

## 9. Conclusion

EDA provides an understanding of the GPS dataset before model training.

The analysis helps identify data quality issues, understand sampling behaviour, examine trip characteristics and visualize spatial trajectories.

These findings support the preprocessing and feature engineering stages of PathSnap.