# PathSnap – Dataset Documentation

## 1. Introduction

PathSnap uses GPS trajectory data and road network information to perform map matching and route reconstruction.

The dataset contains vehicle movement observations and the corresponding road network representation.

The project uses preprocessed candidate feature datasets for machine learning.

## 2. Dataset Organization

The original trip data is organized into individual trip directories.

Each trip can contain the following files:

| File | Description |
|---|---|
| `.track` | GPS observations |
| `.nodes` | Road network node coordinates |
| `.arcs` | Connections between road network nodes |
| `.route` | Ground-truth road arc sequence, when available |

## 3. Dataset Partitions

According to the committed dataset inventory:

| Partition | Number of trips | GPS observations |
|---|---:|---:|
| Training | 80 | 208,417 |
| Student Testing | 10 | 19,160 |
| Total | 90 | 227,577 |

The training partition contains 31,336 ground-truth traversed arcs, while the student testing partition contains 3,477.

## 4. Raw Dataset Fields

### 4.1 Track File

The track file contains three columns:

| Column | Description |
|---|---|
| Longitude | Geographic longitude |
| Latitude | Geographic latitude |
| Time | GPS observation timestamp |

### 4.2 Nodes File

| Column | Description |
|---|---|
| Longitude | Node longitude |
| Latitude | Node latitude |

Each row represents a road network node.

### 4.3 Arcs File

| Column | Description |
|---|---|
| u | Starting node index |
| v | Ending node index |

An arc represents a connection between two road network nodes.

### 4.4 Route File

The route file contains road arc identifiers representing the ground-truth route.

It is used to associate GPS observations with their correct road segments during supervised training and evaluation.

## 5. Preprocessed Feature Datasets

The project stores machine learning datasets under `data/`.

- `train_features.parquet`
- `test_features.parquet`

These contain candidate-level features generated during preprocessing.

The README reports 508,340 candidate rows for training and 191,600 for testing in the documented feature dataset.

## 6. Feature Description

The machine learning pipeline uses 13 features.

| Feature | Description |
|---|---|
| dist_to_segment | Distance from GPS point to candidate road segment |
| dist_to_u | Distance from GPS point to starting node |
| dist_to_v | Distance from GPS point to ending node |
| dist_diff_min | Difference from minimum candidate distance |
| dist_rank | Candidate distance ranking |
| dist_rank_norm | Normalized candidate ranking |
| segment_length | Length of candidate road segment |
| proj_ratio | Position of projected point along segment |
| heading_diff | Difference between GPS and road heading |
| cos_heading_diff | Cosine of heading difference |
| sin_heading_diff | Sine of heading difference |
| gps_speed | Estimated GPS movement speed |
| gps_dt | Time difference between consecutive GPS observations |

## 7. Target Variable

The target variable is `label`.

- `1` – Candidate road segment matches the ground-truth road arc.
- `0` – Candidate road segment does not match the ground-truth road arc.

This makes the problem a binary classification task.

## 8. Data Preprocessing

The preprocessing pipeline performs the following operations:

1. Loads GPS tracks, road nodes and road arcs.
2. Converts geographic coordinates into local metric coordinates.
3. Builds a KD-tree using road segment midpoints.
4. Finds nearby candidate road segments for each GPS observation.
5. Calculates point-to-segment distances.
6. Computes movement and heading features.
7. Ranks candidates by distance.
8. Assigns labels using ground-truth route information when available.
9. Generates a structured feature DataFrame.

## 9. Data Quality

The committed dataset inventory reports:

- Missing values: 0 in training.
- Missing values: 0 in student testing.
- Invalid coordinates: 0 in training.
- Invalid coordinates: 0 in student testing.

## 10. Storage Format

The feature datasets are stored in Parquet format.

Parquet is a column-oriented storage format that supports efficient reading and compression.

## 11. Conclusion

The dataset combines GPS observations, road network topology and candidate-level spatial features.

These features allow machine learning models to estimate which road segment most likely corresponds to each GPS observation.