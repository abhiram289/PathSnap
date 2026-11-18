import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.preprocessing.pipeline import process_single_trip
from src.models.train_models import FEATURE_COLS
from src.hmm.decoder import HMMRouteDecoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("PathSnapAPI")

app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching and Route Reconstruction Server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ASSETS = {}

class TripEvaluationRequest(BaseModel):
    trip_id: str

class TripEvaluationResponse(BaseModel):
    trip_id: str
    num_gps_observations: int
    classifier_top1_accuracy: float
    hmm_reconstructed_accuracy: float
    route_continuity_percent: float
    processing_time_ms: float
    predicted_arcs_sequence: List[int]
    gps_path: List[List[float]]
    reconstructed_path: List[List[float]]
    ground_truth_path: Optional[List[List[float]]] = None

@app.on_event("startup")
def load_assets():
    logger.info("Initializing PathSnap Server & Loading Model Artifacts...")
    model_path = os.path.join("results", "models", "best_tuned_model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join("results", "models", "lightgbm_classifier.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join("results", "models", "logistic_regression.joblib")
    
    if os.path.exists(model_path):
        MODEL_ASSETS['classifier'] = joblib.load(model_path)
        logger.info(f"Loaded classifier: {model_path}")
    else:
        logger.error("No trained classifier found! Please run train_models.py")
        
    MODEL_ASSETS['decoder'] = HMMRouteDecoder(sigma_z=10.0, beta=15.0)
    logger.info("HMM & Viterbi Decoder initialized successfully.")

def find_trip_dir(trip_id: str):
    candidates = [
        os.path.join("Student folder", "student_test", trip_id),
        os.path.join("Student folder", "train", trip_id),
        os.path.join("..", "Student folder", "student_test", trip_id),
        os.path.join("..", "Student folder", "train", trip_id),
        os.path.join(os.path.dirname(__file__), "..", "..", "Student folder", "student_test", trip_id),
        os.path.join(os.path.dirname(__file__), "..", "..", "Student folder", "train", trip_id),
        os.path.join(r"E:\Engg Sem 5\ML\Student folder", "student_test", trip_id),
        os.path.join(r"E:\Engg Sem 5\ML\Student folder", "train", trip_id),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def find_student_test_dir():
    candidates = [
        os.path.join("Student folder", "student_test"),
        os.path.join("..", "Student folder", "student_test"),
        os.path.join(os.path.dirname(__file__), "..", "..", "Student folder", "student_test"),
        r"E:\Engg Sem 5\ML\Student folder\student_test",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

@app.get("/")
def root():
    return {"status": "online", "message": "PathSnap Route Reconstruction API"}

@app.get("/trips")
def list_available_trips():
    """Lists the evaluation trips available on the server."""
    test_dir = find_student_test_dir()
    if test_dir and os.path.exists(test_dir):
        trips = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]
    else:
        trips = []
    return {"available_trips": trips}

@app.post("/evaluate", response_model=TripEvaluationResponse)
def evaluate_trip(req: TripEvaluationRequest):
    t_start = time.time()
    trip_id = req.trip_id
    logger.info(f"Incoming evaluation request received for Trip ID: {trip_id}")
    
    trip_dir = find_trip_dir(trip_id)
    if trip_dir is None:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found on server.")
            
    df_trip = process_single_trip(trip_dir, trip_id, k_query=80, top_k=10, is_training=False)
    if df_trip.empty:
        raise HTTPException(status_code=400, detail=f"No candidates generated for Trip {trip_id}.")
        
    clf = MODEL_ASSETS.get('classifier')
    if clf is None:
        raise HTTPException(status_code=500, detail="Model classifier not loaded.")
        
    df_trip[FEATURE_COLS] = df_trip[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = clf.predict_proba(df_trip[FEATURE_COLS].values)[:, 1]
    df_trip['score'] = scores
    
    # Baseline Top-1 classifier accuracy
    idx_top1 = df_trip.groupby('obs_idx')['score'].idxmax()
    top1_df = df_trip.loc[idx_top1].sort_values('obs_idx').reset_index(drop=True)
    acc_top1 = (top1_df['label'] == 1).mean() * 100.0 if 'label' in top1_df else 0.0
    
    # HMM Viterbi Decoding
    decoder = MODEL_ASSETS.get('decoder')
    hmm_df = decoder.decode_trip(df_trip)
    acc_hmm = (hmm_df['label'] == 1).mean() * 100.0 if 'label' in hmm_df else 0.0
    
    # Route continuity calculation
    ha_arr = hmm_df['predicted_arc'].values
    hu_arr = hmm_df['u'].values
    hv_arr = hmm_df['v'].values
    
    cont_steps = sum(
        ha_arr[i] == ha_arr[i+1] or 
        hv_arr[i] == hu_arr[i+1] or 
        hu_arr[i] == hu_arr[i+1] or 
        hv_arr[i] == hv_arr[i+1]
        for i in range(len(ha_arr) - 1)
    )
    continuity = (cont_steps / max(1, len(ha_arr) - 1)) * 100.0
    
    # Load geometries for Folium map rendering
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    
    track = pd.read_csv(track_path, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_path, sep=r'\s+', header=None, names=['lon', 'lat'])
    arcs = pd.read_csv(arcs_path, sep=r'\s+', header=None, names=['u', 'v'])
    
    gps_coords = track[['lat', 'lon']].values.tolist()
    
    reconstructed_coords = []
    unique_arcs = []
    prev_arc = None
    for a_id in ha_arr:
        if a_id != prev_arc:
            unique_arcs.append(int(a_id))
            prev_arc = a_id
            u = arcs.loc[a_id, 'u']
            v = arcs.loc[a_id, 'v']
            reconstructed_coords.append([nodes.loc[u, 'lat'], nodes.loc[u, 'lon']])
            reconstructed_coords.append([nodes.loc[v, 'lat'], nodes.loc[v, 'lon']])
            
    gt_coords = None
    route_path = os.path.join(trip_dir, f"{trip_id}.route")
    if os.path.exists(route_path):
        route = pd.read_csv(route_path, sep=r'\s+', header=None, names=['arc_id'])
        gt_coords = []
        for r_id in route['arc_id'].values:
            u = arcs.loc[r_id, 'u']
            v = arcs.loc[r_id, 'v']
            gt_coords.append([nodes.loc[u, 'lat'], nodes.loc[u, 'lon']])
            gt_coords.append([nodes.loc[v, 'lat'], nodes.loc[v, 'lon']])
            
    total_time_ms = (time.time() - t_start) * 1000.0
    logger.info(f"Trip {trip_id} inference completed in {total_time_ms:.1f}ms: Top1={acc_top1:.2f}%, HMM={acc_hmm:.2f}%, Continuity={continuity:.2f}%")
    
    return TripEvaluationResponse(
        trip_id=trip_id,
        num_gps_observations=len(track),
        classifier_top1_accuracy=round(acc_top1, 2),
        hmm_reconstructed_accuracy=round(acc_hmm, 2),
        route_continuity_percent=round(continuity, 2),
        processing_time_ms=round(total_time_ms, 1),
        predicted_arcs_sequence=unique_arcs,
        gps_path=gps_coords,
        reconstructed_path=reconstructed_coords,
        ground_truth_path=gt_coords
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
