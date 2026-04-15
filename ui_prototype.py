import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import base64
import joblib

# config
from config import PITCH_TYPES, STRIKE_ZONE, MOVEMENT_SCALE, MOVEMENT_THRESHOLD

CHART_HEIGHT = 600

st.set_page_config(
    page_title="Batter Up!",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    f"""
    <style>
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {{
        max-height: {CHART_HEIGHT - 60}px;
        overflow-y: auto;
        overflow-x: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

################# MODELING #################
# Cache and load pitching model
@st.cache_resource
def load_pitching_model():
    model_a = joblib.load('models/pitch_outcome_model.joblib')
    labeler_a = joblib.load('models/pitch_outcome_labeler.joblib')
    return model_a, labeler_a

# Cache and load batting model
@st.cache_resource
def load_batting_model():
    model_b = joblib.load('models/batted_outcome_model.joblib')
    labeler_b = joblib.load('models/batted_outcome_labler.joblib')
    return model_b, labeler_b

def predict_pitch_outcome(model, labeler, user_inputs):
    
    user_inputs['stand'] = 1 if user_inputs['stand'] == 'R' else 0
    user_inputs['p_throws'] = 1 if user_inputs['p_throws'] == 'R' else 0
    
    df = pd.DataFrame([user_inputs])
    df['pitch_name'] = df['pitch_name'].astype('category')
    
    probs = model.predict_proba(df)[0]
    
    return dict(zip(labeler.classes_, probs))

# Call fns to load in models
batting_model, batting_labeler = load_batting_model()
pitching_model, pitching_labeler = load_pitching_model()
###################################################


def load_svg(svg_path):
    try:
        with open(svg_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return None

def create_viz(plate_x, plate_z, batter_handedness, pfx_x=0, pfx_z=0, release_speed=92):
    fig = go.Figure()
    
    sz_left = STRIKE_ZONE['left']
    sz_right = STRIKE_ZONE['right']
    sz_bottom = STRIKE_ZONE['bottom']
    sz_top = STRIKE_ZONE['top']
    
    # strike zone
    fig.add_shape(
        type="rect",
        x0=sz_left, y0=sz_bottom, x1=sz_right, y1=sz_top,
        line=dict(color="red", width=2),
        fillcolor="rgba(255,0,0,0.05)"
    )
    
    # batter silhouette 
    if batter_handedness == 'R': 
        batter_x = -2.6
        svg_file = "assets/svg/batter_right.svg"
    else:
        batter_x = 2.6
        svg_file = "assets/svg/batter_left.svg"
    
    svg_content = load_svg(svg_file)
    if svg_content:
        svg_b64 = base64.b64encode(svg_content.encode()).decode()
        svg_uri = f"data:image/svg+xml;base64,{svg_b64}"
        sz_height = sz_top - sz_bottom
        sz_width = sz_right - sz_left
        
        batter_height = sz_height * 2.7
        batter_width = sz_width * 6
        batter_y = sz_top + (batter_height - sz_height) * 0.55
        
        fig.add_layout_image(
            dict(
                source=svg_uri,
                xref="x",
                yref="y",
                x=batter_x,
                y=batter_y,
                sizex=batter_width,
                sizey=batter_height,
                xanchor="center",
                yanchor="top",
                opacity=1.0,
                layer="below"
            )
        )

    # ball
    in_zone = (sz_left <= plate_x <= sz_right) and (sz_bottom <= plate_z <= sz_top)
    ball_color = "red" if in_zone else "green"
    ball_edge_color = "white"
    
    # glow that intensifies with velocity
    glow_intensity = (release_speed - 70) / 35  # normalize
    log_intensity = np.log(1 + glow_intensity * 9) / np.log(10)
    glow_size = 30 + (log_intensity * 5)
    base_opacity = 0.01 + (log_intensity * 0.3)
    
    # gradient glow 
    for i, scale in enumerate([1.0, 0.75, 0.5]):
        fig.add_trace(go.Scatter(
            x=[plate_x],
            y=[plate_z],
            mode='markers',
            marker=dict(
                size=glow_size * scale,
                color='red',
                opacity=base_opacity * (1.5 - i * 0.3),
                symbol='circle'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.add_trace(go.Scatter(
        x=[plate_x],
        y=[plate_z],
        mode='markers',
        marker=dict(
            size=20,
            color=ball_color,
            line=dict(color=ball_edge_color, width=3),
            symbol='circle'
        ),
        name='Location',
        showlegend=False,
        hovertemplate=f"""<b>Ball Location</b><br>X: {plate_x:.2f} ft<br>Z: {plate_z:.2f} ft<br>{'Strike' if in_zone else 'Ball'}<br><br><b>Movement</b><br>H: {pfx_x:+.1f}″<br>V: {pfx_z:+.1f}″<br>Speed: {release_speed:.0f} mph<extra></extra>"""
    ))
    
    # movement vectors
    pfx_x_ft = (pfx_x / 12.0) * MOVEMENT_SCALE
    pfx_z_ft = (pfx_z / 12.0) * MOVEMENT_SCALE
    
    if abs(pfx_x) > MOVEMENT_THRESHOLD or abs(pfx_z) > MOVEMENT_THRESHOLD:
        fig.add_annotation(
            x=plate_x, y=plate_z,
            ax=plate_x + pfx_x_ft, ay=plate_z + pfx_z_ft,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=1, arrowsize=1,
            arrowwidth=2, arrowcolor='rgba(255,255,255,0.8)',
            text=""
        )
    
    # home plate SVG  
    homeplate_svg = load_svg("assets/svg/homeplate.svg")
    if homeplate_svg:
        homeplate_b64 = base64.b64encode(homeplate_svg.encode()).decode()
        homeplate_uri = f"data:image/svg+xml;base64,{homeplate_b64}"
        
        plate_width = 2
        plate_depth = 0.85
        
        fig.add_layout_image(
            dict(
                source=homeplate_uri,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=plate_width,
                sizey=plate_depth,
                xanchor="center",
                yanchor="bottom",
                opacity=1.0,
                layer="below"
            )
        )
    
    fig.update_layout(
        title={
            'text': "Catcher POV",
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis=dict(
            range=[-4, 4],
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='rgba(128,128,128,0.3)',
            title="",
            showticklabels=False,
            fixedrange=True 
        ),
        yaxis=dict(
            range=[-1, 5.5],
            showgrid=True,
            gridcolor='rgba(128, 128,128,0.2)',
            zeroline=False,
            title="",
            showticklabels=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),
        plot_bgcolor='rgba(96, 147, 93,0.7)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=CHART_HEIGHT,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

st.title("Batter Up - MLB Pitch Outcome Predictor")
col_controls, col_viz = st.columns([1, 2])

# ================== CONTROL PANEL ==================
with col_controls:
    with st.expander("Pitch Controls", expanded=True):

        st.header("Pitch Controls")
        
        st.subheader("Pitch Type")
        pitch_name = st.selectbox(
            "Select Pitch",
            options=PITCH_TYPES,
            index=0
        )
        
        st.subheader("Pitch Location")
        plate_x = st.slider(
            "Horizontal Position (left/right)",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            help="Position in feet from center of plate. Negative = left, Positive = right (from catcher's view)"   # make sure this isn't backwards
        )
        
        plate_z = st.slider(
            "Vertical Position (up/down)",
            min_value=0.5,
            max_value=5.0,
            value=2.5,
            step=0.05,
            help="Height in feet from ground"
        )
        
        st.subheader("Pitch Characteristics")
        release_speed = st.slider(
            "Release Speed (mph)",
            min_value=70.0, # this should come from the bounds of tehe dataset
            max_value=105.0,
            value=92.0,
            step=0.5
        )
        
        release_spin_rate = st.slider(
            "Spin Rate (rpm)",
            min_value=1500,
            max_value=3500,
            value=2200,
            step=50
        )
        
        pfx_x = st.slider(
            "Horizontal Movement (inches)",
            min_value=-20.0,
            max_value=20.0,
            value=0.0,
            step=0.5,
            help="Movement in inches from catcher's perspective"
        )
        
        pfx_z = st.slider(
            "Vertical Movement (inches)",
            min_value=-25.0,
            max_value=25.0,
            value=0.0,
            step=0.5,
            help="Movement in inches from catcher's perspective"
        )
        
        release_extension = st.slider(
            "Release Extension (feet)",
            min_value=5.5,
            max_value=8.0,
            value=6.0,
            step=0.05,
            help="Distance from pitching rubber at release point"
        )
        
        arm_angle = st.slider(
            "Arm Angle (degrees)",
            min_value=0,
            max_value=90,
            value=45,
            step=1,
            help="Release arm angle"
        )
        
        st.subheader("Game Context")
        
        col_count1, col_count2 = st.columns(2)
        with col_count1:
            balls = st.selectbox("Balls", options=[0, 1, 2, 3], index=0)
        with col_count2:
            strikes = st.selectbox("Strikes", options=[0, 1, 2], index=0)
        
        col_hand1, col_hand2 = st.columns(2)
        with col_hand1:
            stand = st.selectbox("Batter", options=['R', 'L'], index=0, format_func=lambda x: f"Right" if x == 'R' else "Left")
        with col_hand2:
            p_throws = st.selectbox("Pitcher", options=['R', 'L'], index=0, format_func=lambda x: f"Right" if x == 'R' else "Left")
        outs_when_up = st.selectbox("Outs", options=[0, 1, 2], index=0)

with col_viz:
    catcher_fig = create_viz(plate_x, plate_z, stand, pfx_x, pfx_z, release_speed)
    st.plotly_chart(catcher_fig, use_container_width=True)
    
    st.subheader("Outcome Probabilities")

    current_pitch_data = {
        'pitch_name': pitch_name,
        'plate_x': plate_x,
        'plate_z': plate_z,
        'release_speed': release_speed,
        'release_spin_rate': release_spin_rate,
        'pfx_x': pfx_x,
        'pfx_z': pfx_z,
        'balls': balls,
        'strikes': strikes,
        'stand': stand, 
        'p_throws': p_throws,
        'release_extension': release_extension,
        'arm_angle': arm_angle
    }
    
    #  Pitching pred
    results = predict_pitch_outcome(pitching_model, pitching_labeler, current_pitch_data)

    outcome_cols = st.columns(4)
    with outcome_cols[0]:
        st.metric("Ball", f"{results.get('ball', 0) * 100:.1f}%")
    with outcome_cols[1]:
        st.metric("Strike", f"{results.get('strike', 0) * 100:.1f}%")
    with outcome_cols[2]:
        st.metric("Foul", f"{results.get('foul_ball', 0) * 100:.1f}%")
    with outcome_cols[3]:
        st.metric("In Play", f"{results.get('in_play', 0) * 100:.1f}%")

st.markdown("---")
st.caption("CSE 6242 Project | Data from MLB Statcast (2021-2025)")
