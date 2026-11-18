import os
import time
import numpy as np
import pandas as pd
import networkx as nx

class HMMRouteDecoder:
    def __init__(self, sigma_z=10.0, beta=15.0):
        self.sigma_z = sigma_z
        self.beta = beta
        
    def decode_trip(self, df_trip_candidates, graph=None):
        """
        df_trip_candidates: DataFrame of candidates for ONE trip, sorted by obs_idx
        Contains: ['obs_idx', 'arc_id', 'u', 'v', 'dist_to_segment', 'score']
        """
        obs_groups = df_trip_candidates.groupby('obs_idx')
        obs_indices = sorted(obs_groups.groups.keys())
        T = len(obs_indices)
        if T == 0:
            return []
            
        trellis = []
        backpointers = []
        candidates_by_step = []
        
        # Build 1-hop outgoing adjacency if graph is provided
        # or extract directly from arc (u, v) pairs
        for t_step, o_idx in enumerate(obs_indices):
            cands = obs_groups.get_group(o_idx).reset_index(drop=True)
            candidates_by_step.append(cands)
            K = len(cands)
            
            # Emission probabilities: combine classifier confidence and spatial distance
            scores = np.clip(cands['score'].values, 1e-6, 1.0 - 1e-6)
            dists = cands['dist_to_segment'].values
            log_emissions = np.log(scores) - (dists / (self.sigma_z + 1e-5))
            
            if t_step == 0:
                # Prior: uniform or purely emission
                trellis.append(log_emissions)
                backpointers.append(np.zeros(K, dtype=int))
            else:
                prev_V = trellis[t_step - 1]
                prev_cands = candidates_by_step[t_step - 1]
                prev_K = len(prev_cands)
                
                curr_V = np.zeros(K)
                curr_bp = np.zeros(K, dtype=int)
                
                prev_u = prev_cands['u'].values
                prev_v = prev_cands['v'].values
                prev_arcs = prev_cands['arc_id'].values
                
                curr_u = cands['u'].values
                curr_v = cands['v'].values
                curr_arcs = cands['arc_id'].values
                
                # Compute transition log probabilities
                for k in range(K):
                    c_a, c_u_node, c_v_node = curr_arcs[k], curr_u[k], curr_v[k]
                    
                    trans_log_probs = np.full(prev_K, -12.0) # penalty for disconnected transitions
                    
                    for j in range(prev_K):
                        p_a, p_u_node, p_v_node = prev_arcs[j], prev_u[j], prev_v[j]
                        
                        if c_a == p_a:
                            # Remained on same arc
                            trans_log_probs[j] = np.log(0.60)
                        elif p_v_node == c_u_node:
                            # Direct 1-hop topological road connection: prev_arc -> curr_arc
                            trans_log_probs[j] = np.log(0.35)
                        elif p_u_node == c_u_node or p_v_node == c_v_node:
                            # Intersection neighbors or bidirectional road
                            trans_log_probs[j] = np.log(0.04)
                        else:
                            # Small probability to allow recovery from GPS jumps
                            trans_log_probs[j] = -10.0
                            
                    total_probs = prev_V + trans_log_probs
                    best_prev = np.argmax(total_probs)
                    curr_V[k] = total_probs[best_prev] + log_emissions[k]
                    curr_bp[k] = best_prev
                    
                trellis.append(curr_V)
                backpointers.append(curr_bp)
                
        # Backtrack optimal sequence of states
        best_last_k = np.argmax(trellis[T - 1])
        decoded_path = [best_last_k]
        
        for t_step in range(T - 1, 0, -1):
            best_prev = backpointers[t_step][decoded_path[-1]]
            decoded_path.append(best_prev)
            
        decoded_path.reverse()
        
        # Extract chosen candidate arc records
        decoded_results = []
        for t_step, cand_idx in enumerate(decoded_path):
            chosen_row = candidates_by_step[t_step].iloc[cand_idx]
            decoded_results.append({
                'obs_idx': obs_indices[t_step],
                'predicted_arc': chosen_row['arc_id'],
                'u': chosen_row['u'],
                'v': chosen_row['v'],
                'score': chosen_row['score'],
                'dist_to_segment': chosen_row['dist_to_segment'],
                'label': chosen_row.get('label', 0)
            })
            
        return pd.DataFrame(decoded_results)

def evaluate_hmm_reconstruction():
    print("=" * 70)
    print("PATHSNAP: HMM & VITERBI ROUTE RECONSTRUCTION BENCHMARK")
    print("=" * 70)
    
    test_path = os.path.join("data", "test_features.csv")
    test_df = pd.read_csv(test_path)
    
    FEATURE_COLS = [
        'dist_to_segment', 'dist_to_u', 'dist_to_v', 'dist_diff_min',
        'dist_rank', 'dist_rank_norm', 'segment_length', 'proj_ratio',
        'heading_diff', 'cos_heading_diff', 'sin_heading_diff',
        'gps_speed', 'gps_dt'
    ]
    
    # Load best model
    model_path = os.path.join("results", "models", "best_tuned_model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join("results", "models", "lightgbm_classifier.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join("results", "models", "logistic_regression.joblib")
    import joblib
    clf = joblib.load(model_path)
    print(f"Loaded model for HMM emissions: {os.path.basename(model_path)}")
    
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_df['score'] = clf.predict_proba(test_df[FEATURE_COLS].values)[:, 1]
    
    decoder = HMMRouteDecoder(sigma_z=10.0, beta=15.0)
    
    test_trips = sorted(test_df['trip_id'].unique())
    print(f"Evaluating across {len(test_trips)} student testing trips...")
    
    trip_summaries = []
    
    for tid in test_trips:
        df_trip = test_df[test_df['trip_id'] == tid].copy()
        
        # 1. Classifier-Only Top-1
        idx_top1 = df_trip.groupby('obs_idx')['score'].idxmax()
        top1_df = df_trip.loc[idx_top1].sort_values('obs_idx').reset_index(drop=True)
        acc_top1 = (top1_df['label'] == 1).mean()
        
        # Continuity of Classifier-Only
        u_arr, v_arr, a_arr = top1_df['u'].values, top1_df['v'].values, top1_df['arc_id'].values
        cont_top1 = sum(a_arr[i] == a_arr[i+1] or v_arr[i] == u_arr[i+1] or u_arr[i] == u_arr[i+1] or v_arr[i] == v_arr[i+1] for i in range(len(a_arr)-1)) / max(1, len(a_arr)-1)
        
        # 2. Classifier + HMM Viterbi Decoding
        t_dec0 = time.time()
        hmm_df = decoder.decode_trip(df_trip)
        dec_time = time.time() - t_dec0
        
        acc_hmm = (hmm_df['label'] == 1).mean()
        hu_arr, hv_arr, ha_arr = hmm_df['u'].values, hmm_df['v'].values, hmm_df['predicted_arc'].values
        cont_hmm = sum(ha_arr[i] == ha_arr[i+1] or hv_arr[i] == hu_arr[i+1] or hu_arr[i] == hu_arr[i+1] or hv_arr[i] == hv_arr[i+1] for i in range(len(ha_arr)-1)) / max(1, len(ha_arr)-1)
        
        trip_summaries.append({
            'trip_id': tid,
            'points': len(top1_df),
            'Classifier_Top1_Acc': round(acc_top1 * 100, 2),
            'HMM_Viterbi_Acc': round(acc_hmm * 100, 2),
            'Classifier_Continuity': round(cont_top1 * 100, 2),
            'HMM_Continuity': round(cont_hmm * 100, 2),
            'Decode_Time_s': round(dec_time, 3)
        })
        print(f"Trip {str(tid):8s} | Points: {len(top1_df):4d} | Acc: Top1={acc_top1*100:5.2f}% -> HMM={acc_hmm*100:5.2f}% | Continuity: Top1={cont_top1*100:5.2f}% -> HMM={cont_hmm*100:5.2f}%")
        
    summary_df = pd.DataFrame(trip_summaries)
    out_csv = os.path.join("results", "metrics", "hmm_vs_classifier_comparison.csv")
    summary_df.to_csv(out_csv, index=False)
    
    print("\n--- OVERALL HMM VS CLASSIFIER-ONLY BENCHMARK ---")
    mean_top1_acc = summary_df['Classifier_Top1_Acc'].mean()
    mean_hmm_acc = summary_df['HMM_Viterbi_Acc'].mean()
    mean_top1_cont = summary_df['Classifier_Continuity'].mean()
    mean_hmm_cont = summary_df['HMM_Continuity'].mean()
    
    print(f"Mean Observation Accuracy: Classifier-Only = {mean_top1_acc:.2f}% | HMM Viterbi = {mean_hmm_acc:.2f}%")
    print(f"Mean Route Continuity:   Classifier-Only = {mean_top1_cont:.2f}% | HMM Viterbi = {mean_hmm_cont:.2f}%")
    print(f"Saved comparison to: {out_csv}")

if __name__ == "__main__":
    evaluate_hmm_reconstruction()
