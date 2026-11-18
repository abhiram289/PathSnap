import os
import time
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def project_to_meters(lon, lat, lon0, lat0):
    R = 6371000.0
    lat0_rad = np.radians(lat0)
    x = (np.radians(lon - lon0)) * R * np.cos(lat0_rad)
    y = (np.radians(lat - lat0)) * R
    return x, y

def point_to_segment_dist_and_proj(px, py, ux, uy, vx, vy):
    dx = vx - ux
    dy = vy - uy
    seg_len_sq = dx*dx + dy*dy
    if seg_len_sq < 1e-6:
        dist = np.hypot(px - ux, py - uy)
        return dist, 0.0, 0.0
    
    seg_len = np.sqrt(seg_len_sq)
    t = ((px - ux)*dx + (py - uy)*dy) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_x = ux + t_clamped * dx
    proj_y = uy + t_clamped * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t_clamped, seg_len

def calculate_bearing(x1, y1, x2, y2):
    # Angle in radians from vector (x1, y1) -> (x2, y2)
    return np.arctan2(y2 - y1, x2 - x1)

def angle_difference(a1, a2):
    # Shortest angular difference between two angles in radians [-pi, pi]
    diff = (a1 - a2 + np.pi) % (2 * np.pi) - np.pi
    return np.abs(diff)

def process_single_trip(trip_dir, trip_id, k_query=80, top_k=10, is_training=True):
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    
    track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
    
    has_route = os.path.exists(route_path)
    route_arcs = None
    if has_route:
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        route_arcs = route['arc_id'].values
        
    lon0 = track['lon'].mean()
    lat0 = track['lat'].mean()
    
    node_x, node_y = project_to_meters(nodes['lon'].values, nodes['lat'].values, lon0, lat0)
    gps_x, gps_y = project_to_meters(track['lon'].values, track['lat'].values, lon0, lat0)
    gps_time = track['time'].values
    
    arc_u = arcs['u'].values
    arc_v = arcs['v'].values
    ux = node_x[arc_u]
    uy = node_y[arc_u]
    vx = node_x[arc_v]
    vy = node_y[arc_v]
    
    mid_x = (ux + vx) / 2.0
    mid_y = (uy + vy) / 2.0
    arc_azimuth = np.arctan2(vy - uy, vx - ux)
    
    # KDTree for fast spatial nearest midpoint retrieval
    tree = cKDTree(np.column_stack((mid_x, mid_y)))
    
    # Ground truth mapping if route is available
    gt_arcs = None
    if has_route and route_arcs is not None:
        r_u = node_x[arc_u[route_arcs]]
        r_y = node_y[arc_u[route_arcs]]
        r_vx = node_x[arc_v[route_arcs]]
        r_vy = node_y[arc_v[route_arcs]]
        
        gt_arcs = []
        curr_idx = 0
        for i in range(len(track)):
            px, py = gps_x[i], gps_y[i]
            best_d = float('inf')
            best_a = route_arcs[curr_idx]
            best_idx = curr_idx
            w_start = max(0, curr_idx - 2)
            w_end = min(len(route_arcs), curr_idx + 15)
            for r_i in range(w_start, w_end):
                d, _, _ = point_to_segment_dist_and_proj(px, py, r_u[r_i], r_y[r_i], r_vx[r_i], r_vy[r_i])
                if d < best_d:
                    best_d = d
                    best_a = route_arcs[r_i]
                    best_idx = r_i
            curr_idx = best_idx
            gt_arcs.append(best_a)
        gt_arcs = np.array(gt_arcs)
        
    # Query tree
    _, indices = tree.query(np.column_stack((gps_x, gps_y)), k=k_query)
    
    # Extract features for all candidates
    features_list = []
    
    # Precompute GPS trajectory velocity & heading
    num_pts = len(track)
    gps_headings = np.zeros(num_pts)
    gps_speeds = np.zeros(num_pts)
    gps_dt = np.zeros(num_pts)
    
    for i in range(num_pts):
        if i == 0 and num_pts > 1:
            dx = gps_x[1] - gps_x[0]
            dy = gps_y[1] - gps_y[0]
            dt = max(1e-3, gps_time[1] - gps_time[0])
        elif i > 0:
            dx = gps_x[i] - gps_x[i-1]
            dy = gps_y[i] - gps_y[i-1]
            dt = max(1e-3, gps_time[i] - gps_time[i-1])
        else:
            dx, dy, dt = 0.0, 0.0, 1.0
            
        dist = np.hypot(dx, dy)
        gps_headings[i] = np.arctan2(dy, dx)
        gps_speeds[i] = dist / dt
        gps_dt[i] = dt
        
    for i in range(num_pts):
        px, py = gps_x[i], gps_y[i]
        p_hd = gps_headings[i]
        p_spd = gps_speeds[i]
        p_dt = gps_dt[i]
        gt_a = gt_arcs[i] if gt_arcs is not None else -1
        
        cand_indices = indices[i]
        # Compute exact segment distances
        cand_records = []
        for c_idx in cand_indices:
            d, t, slen = point_to_segment_dist_and_proj(px, py, ux[c_idx], uy[c_idx], vx[c_idx], vy[c_idx])
            cand_records.append((d, t, slen, c_idx))
            
        cand_records.sort(key=lambda x: x[0])
        top_records = cand_records[:top_k]
        
        min_d = top_records[0][0] if len(top_records) > 0 else 0.0
        
        for rank, (dist, t_proj, seg_len, c_idx) in enumerate(top_records):
            c_u, c_v = arc_u[c_idx], arc_v[c_idx]
            dist_u = np.hypot(px - ux[c_idx], py - uy[c_idx])
            dist_v = np.hypot(px - vx[c_idx], py - vy[c_idx])
            
            # Heading difference
            h_diff = angle_difference(p_hd, arc_azimuth[c_idx])
            cos_h_diff = np.cos(h_diff)
            sin_h_diff = np.sin(h_diff)
            
            label = 1 if (gt_a != -1 and c_idx == gt_a) else 0
            
            features_list.append({
                'trip_id': trip_id,
                'obs_idx': i,
                'arc_id': c_idx,
                'u': c_u,
                'v': c_v,
                'dist_to_segment': dist,
                'dist_to_u': dist_u,
                'dist_to_v': dist_v,
                'dist_diff_min': dist - min_d,
                'dist_rank': rank,
                'dist_rank_norm': rank / float(top_k),
                'segment_length': seg_len,
                'proj_ratio': t_proj,
                'heading_diff': h_diff,
                'cos_heading_diff': cos_h_diff,
                'sin_heading_diff': sin_h_diff,
                'gps_speed': p_spd,
                'gps_dt': p_dt,
                'label': label
            })
            
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    t0 = time.time()
    print("Testing processing on trip 00000001...")
    df = process_single_trip(os.path.join("Student folder", "train", "00000001"), "00000001", k_query=80, top_k=10)
    print(f"Generated {len(df)} candidate rows in {time.time() - t0:.2f}s")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print("\nFeature sample:")
    print(df.head(3))
