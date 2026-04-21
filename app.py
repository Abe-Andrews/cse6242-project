import streamlit as st

from components.visualization import create_viz
from components.model_utils import load_pitching_model, load_batting_model, predict_pitch_outcome, predict_batted_outcome
from components.controls import render_control_panel
from components.results_display import display_prediction_results, display_batted_results

CHART_HEIGHT = 700

st.set_page_config(
    page_title="Batter Up!",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
    }
    div[data-testid="stExpander"] {
        position: fixed;
        top: 80px;
        left: 20px;
        width: 350px;
        z-index: 999;
        background-color: rgba(17, 17, 17, 0.95);
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        max-height: calc(100vh - 200px);
        overflow-y: auto;
    }
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    div[data-testid="stPlotlyChart"] .js-plotly-plot {
        border-radius: 15px !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# Call fns to load in models
batting_model, batting_labeler = load_batting_model()
pitching_model, pitching_labeler = load_pitching_model()

# st.title("Batter Up - MLB Pitch Outcome Predictor")

# ================== CONTROL PANEL ==================
inputs = render_control_panel()

# ================== MAIN CONTENT ==================
col_viz, col_results = st.columns([1, 1], vertical_alignment="center")

with col_viz:
    catcher_fig = create_viz(inputs['plate_x'], inputs['plate_z'], inputs['stand'], inputs['pfx_x'], inputs['pfx_z'], inputs['release_speed'], CHART_HEIGHT, inputs['p_throws'])
    st.plotly_chart(catcher_fig, use_container_width=True)

with col_results:
    st.subheader("Outcome Probabilities")
    
    current_pitch_data = {
        'pitch_name': inputs['pitch_name'],
        'plate_x': inputs['plate_x'],
        'plate_z': inputs['plate_z'],
        'release_speed': inputs['release_speed'],
        'release_spin_rate': inputs['release_spin_rate'],
        'pfx_x': inputs['pfx_x'],
        'pfx_z': inputs['pfx_z'],
        'balls': inputs['balls'],
        'strikes': inputs['strikes'],
        'stand': inputs['stand'], 
        'p_throws': inputs['p_throws'],
        'release_extension': inputs['release_extension'],
        'arm_angle': inputs['arm_angle'],
        'outs_when_up': inputs['outs_when_up']
    }
    
    #  Pitching pred
    results = predict_pitch_outcome(pitching_model, pitching_labeler, current_pitch_data)
    display_prediction_results(results)

    if inputs['use_batted_model']:
        st.markdown("---")
        st.subheader("Batted Ball Outcomes")
        batted_inputs = {
            'bb_type': inputs['bb_type'],
            'launch_speed_angle': inputs['launch_speed_angle'],
            'hc_x': inputs['hc_x'],
            'hc_y': inputs['hc_y']
        }
        batted_results = predict_batted_outcome(
            batting_model, batting_labeler, current_pitch_data, batted_inputs
        )
        display_batted_results(batted_results)

st.markdown("---")
st.caption("CSE 6242 Project | Data from MLB Statcast (2021-2025)")
