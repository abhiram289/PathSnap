import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="PathSnap | Distributed Map-Matching & Route Reconstruction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling to enforce 'Inter' font while preserving Streamlit Material Icons
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Inter Font */
    html, body, .stApp, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, label, .stMetric, input, select {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Strict protection for Streamlit Material Symbols & sidebar collapse icon */
    [data-testid="stIconMaterial"],
    [class*="material-symbols"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    button[kind="header"],
    button[kind="header"] *,
    [data-testid="stToolbar"] * {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined", sans-serif !important;
    }

    /* Clean spacing for sidebar header */
    [data-testid="stSidebarHeader"] {
        padding-top: 0.5rem;
    }

    /* Delta indicator color in st.metric */
    [data-testid="stMetricDelta"] svg {
        fill: #10B981 !important;
    }

    /* Prevent page greying out / dimming during reruns */
    .stApp[data-test-script-state="running"] [data-testid="stAppViewBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        opacity: 1 !important;
        filter: none !important;
    }
    
    .map-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 1.8rem;
        padding: 0.6rem 1rem;
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("### Distributed Deployment Config")

api_base_url = st.sidebar.text_input(
    "Laptop 1 (API Server) URL:",
    value="http://10.8.120.126:8000"
)

# Server Health Check
server_online = False
try:
    health_resp = requests.get(f"{api_base_url.rstrip('/')}/", timeout=3)
    if health_resp.status_code == 200:
        st.sidebar.markdown(
            '<div style="color: #4ade80; font-size: 0.9rem; font-weight: 500; margin-bottom: 1.2rem;">'
            '🟢 Laptop 1 API: Connected & Ready</div>',
            unsafe_allow_html=True
        )
        server_online = True
    else:
        st.sidebar.markdown(
            f'<div style="color: #f87171; font-size: 0.9rem; font-weight: 500; margin-bottom: 1.2rem;">'
            f'🔴 Laptop 1 API: Error {health_resp.status_code}</div>',
            unsafe_allow_html=True
        )
except Exception:
    st.sidebar.markdown(
        '<div style="color: #f87171; font-size: 0.9rem; font-weight: 500; margin-bottom: 1.2rem;">'
        '🔴 Laptop 1 API: Disconnected / Unreachable</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("### Select Evaluation Trip")

available_trips = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094",
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

if server_online:
    try:
        trips_resp = requests.get(f"{api_base_url.rstrip('/')}/trips", timeout=3)
        if trips_resp.status_code == 200:
            server_trips = trips_resp.json().get("available_trips", [])
            if server_trips:
                available_trips = server_trips
    except Exception:
        pass

default_idx = available_trips.index("00000040") if "00000040" in available_trips else 0

selected_trip = st.sidebar.selectbox(
    "Reserved Evaluation Trip ID:",
    options=available_trips,
    index=default_idx
)

run_button = st.sidebar.button(
    "Reconstruct & Match Route",
    type="primary",
    use_container_width=True
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("**System Architecture:**")
st.sidebar.markdown(
    "• Laptop 1: FastAPI ML Classifier & HMM Viterbi Decoder<br>"
    "• Laptop 2: Interactive Streamlit Dashboard (Evaluator UI)",
    unsafe_allow_html=True
)

# ----------------- MAIN PANEL -----------------
st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")

# Check if session has previous or current inference data
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if run_button:
    if not server_online:
        st.error("Cannot run evaluation: API server is unreachable. Please verify Laptop 1 is running `uvicorn src.api.main:app`.")
    else:
        with st.spinner(f"Requesting inference from Laptop 1 for Trip {selected_trip}..."):
            try:
                payload = {"trip_id": selected_trip}
                res = requests.post(f"{api_base_url.rstrip('/')}/evaluate", json=payload, timeout=60)
                if res.status_code == 200:
                    st.session_state["last_result"] = res.json()
                else:
                    st.error(f"Inference failed with status {res.status_code}: {res.text}")
            except Exception as ex:
                st.error(f"Failed to communicate with API server: {ex}")

# Render Results
data = st.session_state.get("last_result")

if data is not None:
    matched_trip = data.get("trip_id", selected_trip)
    proc_time = data.get("processing_time_ms", 0.0)
    
    st.success(f"Trip {matched_trip} successfully matched in {proc_time:.1f} ms!")
    
    # 4 Native Metric Columns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label="GPS Observations",
        value=f"{data['num_gps_observations']:,}"
    )
    c2.metric(
        label="Classifier Top-1 Acc",
        value=f"{data['classifier_top1_accuracy']:.2f}%"
    )
    delta_val = data['hmm_reconstructed_accuracy'] - data['classifier_top1_accuracy']
    c3.metric(
        label="HMM Reconstructed Acc",
        value=f"{data['hmm_reconstructed_accuracy']:.2f}%",
        delta=f"↑ {delta_val:.2f}%" if delta_val > 0 else (f"{delta_val:.2f}%" if delta_val < 0 else None)
    )
    c4.metric(
        label="Route Continuity",
        value=f"{data['route_continuity_percent']:.2f}%"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Interactive Trajectory & Route Reconstruction Map")
    
    # High-contrast visual legend
    st.markdown("""
    <div class="map-legend">
        <div class="legend-item">
            <span style="color: #38BDF8; font-weight: 700; font-size: 1.1rem;">● - -</span>
            <span><b>Raw GPS Observations:</b> Noisy sensor coordinates (Cyan dots / dashed)</span>
        </div>
        <div class="legend-item">
            <span style="color: #EF4444; font-weight: 700; font-size: 1.1rem;">━━━</span>
            <span><b>HMM Generated Route:</b> Matched topological trajectory (Vibrant Red line)</span>
        </div>
        <div class="legend-item">
            <span style="color: #10B981; font-weight: 700; font-size: 1.1rem;">━━━</span>
            <span><b>Ground Truth Reference:</b> Actual vehicle road path (Emerald Green)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    gps_coords = data.get("gps_path", [])
    recon_coords = data.get("reconstructed_path", [])
    gt_coords = data.get("ground_truth_path", None)
    
    if gps_coords:
        center_lat = sum(p[0] for p in gps_coords) / len(gps_coords)
        center_lon = sum(p[1] for p in gps_coords) / len(gps_coords)
        
        # Free OpenStreetMap tiles without API key restrictions or watermarks
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap"
        )
        
        # Layer 1 (Bottom): Raw GPS Observations (Cyan dashed path + subtle dots)
        gps_group = folium.FeatureGroup(name="Raw GPS Observations (Cyan)", show=True)
        folium.PolyLine(
            gps_coords,
            color="#0284C7",
            weight=2,
            opacity=0.6,
            dash_array="5, 5",
            tooltip="Raw GPS Observation Sequence"
        ).add_to(gps_group)
        for pt in gps_coords:
            folium.CircleMarker(
                location=pt,
                radius=2.5,
                color="#0284C7",
                fill=True,
                fill_color="#38BDF8",
                fill_opacity=0.5,
                tooltip="Noisy GPS observation"
            ).add_to(gps_group)
        gps_group.add_to(m)

        # Layer 2 (Middle): Ground Truth Route (Emerald green)
        if gt_coords:
            gt_group = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
            folium.PolyLine(
                gt_coords,
                color="#10B981",
                weight=6,
                opacity=0.85,
                tooltip="Ground Truth Route (Surveyed Road Network)"
            ).add_to(gt_group)
            gt_group.add_to(m)
            
        # Layer 3 (TOP): HMM Reconstructed Trajectory (Vibrant Red/Coral on top)
        if recon_coords:
            recon_group = folium.FeatureGroup(name="HMM Reconstructed Route (Red)", show=True)
            folium.PolyLine(
                recon_coords,
                color="#EF4444",
                weight=4.5,
                opacity=0.95,
                tooltip="HMM + Viterbi Reconstructed Route"
            ).add_to(recon_group)
            recon_group.add_to(m)
        
        # Start and End Markers
        folium.Marker(
            location=gps_coords[0],
            popup="Trip Start",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        folium.Marker(
            location=gps_coords[-1],
            popup="Trip End",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
        
        folium.LayerControl(position="topright", collapsed=False).add_to(m)
        st_folium(m, width="100%", height=550)
        
        with st.expander("Reconstructed Arc Sequence (Topological Road Network IDs)"):
            arcs_list = data.get("predicted_arcs_sequence", [])
            st.write(f"**Total Traversed Arcs:** {len(arcs_list)}")
            st.code(", ".join(str(a) for a in arcs_list))
else:
    st.info("Select an evaluation trip from the sidebar and click **'Reconstruct & Match Route'** to run live distributed inference.")
