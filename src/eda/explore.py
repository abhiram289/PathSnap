import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium import plugins
import networkx as nx

def analyze_trip(trip_dir, trip_id, has_route=True):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    # Track
    track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    num_points = len(track)
    dt = track['time'].diff().dropna()
    median_dt = dt.median() if len(dt) > 0 else 0
    max_dt = dt.max() if len(dt) > 0 else 0
    num_gaps = (dt > 2.0).sum() if len(dt) > 0 else 0
    duration = track['time'].max() - track['time'].min() if len(track) > 0 else 0
    
    # Missing / Invalid checks
    missing_pts = track[['lon', 'lat', 'time']].isna().sum().sum()
    invalid_coords = ((track['lon'] < -180) | (track['lon'] > 180) | 
                      (track['lat'] < -90) | (track['lat'] > 90)).sum()
    
    # Extents
    min_lon, max_lon = track['lon'].min(), track['lon'].max()
    min_lat, max_lat = track['lat'].min(), track['lat'].max()
    
    # Nodes & Arcs
    nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
    num_nodes = len(nodes)
    num_arcs = len(arcs)
    
    # Route
    num_route_arcs = 0
    if has_route and os.path.exists(route_path):
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        num_route_arcs = len(route)
        
    return {
        'trip_id': trip_id,
        'num_points': num_points,
        'duration_s': duration,
        'median_dt_s': median_dt,
        'max_dt_s': max_dt,
        'num_gaps_gt_2s': num_gaps,
        'missing_pts': missing_pts,
        'invalid_coords': invalid_coords,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'num_nodes': num_nodes,
        'num_arcs': num_arcs,
        'num_route_arcs': num_route_arcs
    }

def run_dataset_exploration():
    print("=" * 60)
    print("PATHSNAP: COMPREHENSIVE DATASET EXPLORATION & INVENTORY")
    print("=" * 60)
    
    train_root = os.path.join("Student folder", "train")
    test_root = os.path.join("Student folder", "student_test")
    
    train_trip_ids = sorted(os.listdir(train_root))
    test_trip_ids = sorted(os.listdir(test_root))
    
    print(f"Scanning {len(train_trip_ids)} training trips...")
    train_records = []
    for tid in train_trip_ids:
        tdir = os.path.join(train_root, tid)
        if os.path.isdir(tdir):
            train_records.append(analyze_trip(tdir, tid, has_route=True))
            
    print(f"Scanning {len(test_trip_ids)} student-testing trips...")
    test_records = []
    for tid in test_trip_ids:
        tdir = os.path.join(test_root, tid)
        if os.path.isdir(tdir):
            test_records.append(analyze_trip(tdir, tid, has_route=True))
            
    df_train = pd.DataFrame(train_records)
    df_test = pd.DataFrame(test_records)
    
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    os.makedirs(os.path.join("results", "maps"), exist_ok=True)
    
    df_train.to_csv(os.path.join("results", "metrics", "trip_summary_train.csv"), index=False)
    df_test.to_csv(os.path.join("results", "metrics", "trip_summary_test.csv"), index=False)
    
    inventory_summary = pd.DataFrame([
        {
            'Partition': 'Training',
            'Trip Count': len(df_train),
            'Total GPS Points': df_train['num_points'].sum(),
            'Mean Points/Trip': df_train['num_points'].mean(),
            'Min Points/Trip': df_train['num_points'].min(),
            'Max Points/Trip': df_train['num_points'].max(),
            'Total Traversed Arcs': df_train['num_route_arcs'].sum(),
            'Avg Duration (s)': df_train['duration_s'].mean(),
            'Total Missing Values': df_train['missing_pts'].sum(),
            'Total Invalid Coords': df_train['invalid_coords'].sum()
        },
        {
            'Partition': 'Student Testing',
            'Trip Count': len(df_test),
            'Total GPS Points': df_test['num_points'].sum(),
            'Mean Points/Trip': df_test['num_points'].mean(),
            'Min Points/Trip': df_test['num_points'].min(),
            'Max Points/Trip': df_test['num_points'].max(),
            'Total Traversed Arcs': df_test['num_route_arcs'].sum(),
            'Avg Duration (s)': df_test['duration_s'].mean(),
            'Total Missing Values': df_test['missing_pts'].sum(),
            'Total Invalid Coords': df_test['invalid_coords'].sum()
        }
    ])
    inventory_summary.to_csv(os.path.join("results", "metrics", "dataset_inventory.csv"), index=False)
    print("\n--- DATASET INVENTORY SUMMARY ---")
    print(inventory_summary.to_string(index=False))
    
    # -------------------------------------------------------------
    # EDA Plots
    # -------------------------------------------------------------
    print("\nGenerating EDA visualization figures...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # Figure 1: Distributions of GPS Points and Route Lengths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_train['num_points'], bins=20, color='#1f77b4', edgecolor='black', alpha=0.8)
    axes[0].set_title('Distribution of GPS Observations per Trip (Train)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Number of GPS Points', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    
    axes[1].hist(df_train['num_route_arcs'], bins=20, color='#2ca02c', edgecolor='black', alpha=0.8)
    axes[1].set_title('Distribution of Ground-Truth Route Arcs per Trip (Train)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Number of Route Arcs', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    plt.tight_layout()
    fig_path1 = os.path.join("results", "figures", "eda_trip_lengths_and_points.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"Saved: {fig_path1}")
    
    # Figure 2: Sampling Interval Analysis
    sample_trip_dir = os.path.join(train_root, "00000001")
    track_s = pd.read_csv(os.path.join(sample_trip_dir, "00000001.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    dt_s = track_s['time'].diff().dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(track_s['time'], track_s['lat'], label='Latitude over time', color='#d62728', lw=1.5)
    axes[0].set_title('GPS Latitude Progression over Time (Trip 00000001)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0].set_ylabel('Latitude (°N)', fontsize=11)
    axes[0].legend()
    
    # Interval counts
    val_counts = dt_s.value_counts().sort_index()
    val_counts_capped = val_counts[val_counts.index <= 5]
    axes[1].bar(val_counts_capped.index.astype(str), val_counts_capped.values, color='#9467bd', edgecolor='black', alpha=0.8)
    axes[1].set_title('GPS Sampling Interval Frequency (Trip 00000001)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time Delta (Δt in seconds)', fontsize=11)
    axes[1].set_ylabel('Observation Count', fontsize=11)
    plt.tight_layout()
    fig_path2 = os.path.join("results", "figures", "eda_sampling_intervals.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"Saved: {fig_path2}")
    
    # -------------------------------------------------------------
    # Interactive Folium Map Generation
    # -------------------------------------------------------------
    print("\nGenerating interactive Folium visualization maps...")
    generate_folium_map(sample_trip_dir, "00000001", os.path.join("results", "maps", "interactive_map_trip_00000001.html"))
    
    test_sample_dir = os.path.join(test_root, "00000000")
    generate_folium_map(test_sample_dir, "00000000", os.path.join("results", "maps", "interactive_map_trip_00000000_test.html"))
    
    print("\nEDA Completed Successfully!")

def generate_folium_map(trip_dir, trip_id, out_html_path):
    track = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.arcs"), sep=r'\s+', header=None, names=['u', 'v'])
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    has_route = os.path.exists(route_path)
    if has_route:
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
    
    center_lat = track['lat'].mean()
    center_lon = track['lon'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="CartoDB Voyager"
    )
    
    # Add Esri WorldStreetMap as alternative base layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri',
        name='Esri WorldStreetMap'
    ).add_to(m)
    
    # FeatureGroups for LayerControl
    fg_network = folium.FeatureGroup(name="Local Road Network (Sampled)", show=False)
    fg_route = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    
    # 1. Ground Truth Route
    if has_route:
        route_coords = []
        for a_id in route['arc_id'].values:
            u, v = arcs.loc[a_id, 'u'], arcs.loc[a_id, 'v']
            u_lat, u_lon = nodes.loc[u, 'lat'], nodes.loc[u, 'lon']
            v_lat, v_lon = nodes.loc[v, 'lat'], nodes.loc[v, 'lon']
            folium.PolyLine(
                locations=[[u_lat, u_lon], [v_lat, v_lon]],
                color="#00aa00",
                weight=5,
                opacity=0.8,
                tooltip=f"Route Arc: {a_id}"
            ).add_to(fg_route)
            
    # 2. Raw GPS observations
    for i, row in track.iterrows():
        # Add circle markers for GPS points
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3,
            color="#0066cc",
            fill=True,
            fill_color="#3399ff",
            fill_opacity=0.7,
            popup=f"GPS #{i} | t={row['time']}s | ({row['lat']:.5f}, {row['lon']:.5f})"
        ).add_to(fg_gps)
        
    # GPS trajectory line
    gps_path = track[['lat', 'lon']].values.tolist()
    folium.PolyLine(
        locations=gps_path,
        color="#0044bb",
        weight=2,
        dash_array="5, 5",
        opacity=0.6,
        tooltip="GPS Observed Path"
    ).add_to(fg_gps)
    
    # Add start and end pins
    folium.Marker(
        location=gps_path[0],
        popup=f"Start: Trip {trip_id}",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    folium.Marker(
        location=gps_path[-1],
        popup=f"End: Trip {trip_id}",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)
    
    fg_network.add_to(m)
    fg_route.add_to(m)
    fg_gps.add_to(m)
    folium.LayerControl().add_to(m)
    
    m.save(out_html_path)
    print(f"Generated Folium Map: {out_html_path}")

if __name__ == "__main__":
    run_dataset_exploration()
