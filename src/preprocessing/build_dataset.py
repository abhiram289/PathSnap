import os
import time
import pandas as pd
from pipeline import process_single_trip

def build_datasets(num_train_trips=20):
    print("=" * 60)
    print(f"BUILDING FEATURE MATRICES (Train trips: {num_train_trips}, Test trips: 10)")
    print("=" * 60)
    
    train_root = os.path.join("Student folder", "train")
    test_root = os.path.join("Student folder", "student_test")
    
    all_train_trips = sorted(os.listdir(train_root))
    selected_train = all_train_trips[:num_train_trips]
    test_trips = sorted(os.listdir(test_root))
    
    # 1. Process Training Trips
    print(f"\n[1/2] Processing {len(selected_train)} training trips...")
    train_dfs = []
    t_start = time.time()
    for idx, tid in enumerate(selected_train):
        tdir = os.path.join(train_root, tid)
        t_sub = time.time()
        df_trip = process_single_trip(tdir, tid, k_query=80, top_k=10, is_training=True)
        train_dfs.append(df_trip)
        print(f"  [{idx+1:02d}/{len(selected_train):02d}] Trip {tid}: {len(df_trip)} candidates ({time.time()-t_sub:.2f}s)")
        
    full_train_df = pd.concat(train_dfs, ignore_index=True)
    out_train_path = os.path.join("data", "train_features.csv")
    full_train_df.to_csv(out_train_path, index=False)
    print(f"Saved training features to {out_train_path} ({len(full_train_df)} rows) in {time.time()-t_start:.1f}s")
    
    # 2. Process Student Testing Trips
    print(f"\n[2/2] Processing {len(test_trips)} student testing trips...")
    test_dfs = []
    t_test_start = time.time()
    for idx, tid in enumerate(test_trips):
        tdir = os.path.join(test_root, tid)
        t_sub = time.time()
        df_trip = process_single_trip(tdir, tid, k_query=80, top_k=10, is_training=False)
        test_dfs.append(df_trip)
        print(f"  [{idx+1:02d}/{len(test_trips):02d}] Trip {tid}: {len(df_trip)} candidates ({time.time()-t_sub:.2f}s)")
        
    full_test_df = pd.concat(test_dfs, ignore_index=True)
    out_test_path = os.path.join("data", "test_features.csv")
    full_test_df.to_csv(out_test_path, index=False)
    print(f"Saved testing features to {out_test_path} ({len(full_test_df)} rows) in {time.time()-t_test_start:.1f}s")
    
    print("\nDataset extraction complete!")

if __name__ == "__main__":
    build_datasets(num_train_trips=20)
