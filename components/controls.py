import streamlit as st

from config import PITCH_TYPES, BB_TYPES, LAUNCH_SPEED_ANGLE_LABELS, PITCH_PRESETS


def _apply_preset():
    pitch = st.session_state.get('pitch_select')
    hand = st.session_state.get('p_throws_select', 'R')
    
    if pitch in PITCH_PRESETS and hand in PITCH_PRESETS[pitch]:
        p = PITCH_PRESETS[pitch][hand]
        st.session_state['preset_release_speed'] = p['release_speed']
        st.session_state['preset_pfx_x'] = p['pfx_x']
        st.session_state['preset_pfx_z'] = p['pfx_z']


def render_control_panel():
    if '_presets_initialized' not in st.session_state:
        p = PITCH_PRESETS[PITCH_TYPES[0]]['R']
        st.session_state['pitch_select'] = PITCH_TYPES[0]
        st.session_state['p_throws_select'] = 'R'
        st.session_state['preset_release_speed'] = p['release_speed']
        st.session_state['preset_pfx_x'] = p['pfx_x']
        st.session_state['preset_pfx_z'] = p['pfx_z']
        st.session_state['_presets_initialized'] = True
    with st.expander("Pitch Controls", expanded=False):

        st.header("Pitch Controls")
        
        st.subheader("Pitch Type")
        col_pitch1, col_pitch2 = st.columns(2)
        with col_pitch1:
            pitch_name = st.selectbox(
                "Pitch",
                options=PITCH_TYPES,
                key='pitch_select',
                on_change=_apply_preset
            )
        with col_pitch2:
            p_throws = st.selectbox(
                "Pitcher Hand",
                options=['R', 'L'],
                key='p_throws_select',
                on_change=_apply_preset,
                format_func=lambda x: "Right" if x == 'R' else "Left"
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
            min_value=70.0,
            max_value=105.0,
            step=0.1,
            key='preset_release_speed'
        )
        
        release_spin_rate = st.slider(
            "Spin Rate (rpm)",
            min_value=1300,
            max_value=3500,
            step=50,
            key='preset_release_spin_rate'
        )
        
        pfx_x = st.slider(
            "Horizontal Movement (feet)",
            min_value=-3.0,
            max_value=3.0,
            step=0.05,
            key='preset_pfx_x',
            help="Movement in feet from catcher's perspective"
        )
        
        pfx_z = st.slider(
            "Vertical Movement (feet)",
            min_value=-3.0,
            max_value=3.0,
            step=0.05,
            key='preset_pfx_z',
            help="Movement in feet from catcher's perspective"
        )
        
        release_extension = st.slider(
            "Release Extension (feet)",
            min_value=5.5,
            max_value=8.0,
            step=0.05,
            key='preset_release_extension',
            help="Distance from pitching rubber at release point"
        )
        
        arm_angle = st.slider(
            "Arm Angle (degrees)",
            min_value=0.0,
            max_value=90.0,
            step=0.5,
            key='preset_arm_angle',
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
            p_throws = st.session_state['p_throws_select']
            st.text_input("Pitcher Hand", value="Right" if p_throws == 'R' else "Left", disabled=True)
        outs_when_up = st.selectbox("Outs", options=[0, 1, 2], index=0)

        st.subheader("Contact Metrics")
        use_batted_model = st.toggle(
            "Use Batted Ball Model", value=False,
            help="Include batted ball metrics for a more detailed in-play prediction"
        )

        if use_batted_model:
            bb_type = st.selectbox(
                "Batted Ball Type", options=BB_TYPES, index=0,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            launch_speed_angle = st.selectbox(
                "Contact Quality",
                options=list(LAUNCH_SPEED_ANGLE_LABELS.keys()), index=4,
                format_func=lambda x: f"{x} – {LAUNCH_SPEED_ANGLE_LABELS[x]}"
            )
            hc_x = st.slider(
                "Hit Coordinate X (ft)", min_value=0.0, max_value=250.0, value=125.0, step=1.0,
                help="Horizontal position where ball landed/was fielded"
            )
            hc_y = st.slider(
                "Hit Coordinate Y (ft)", min_value=0.0, max_value=250.0, value=125.0, step=1.0,
                help="Vertical position where ball landed/was fielded"
            )
        else:
            bb_type = launch_speed_angle = hc_x = hc_y = None

    return {
        'pitch_name': pitch_name,
        'plate_x': plate_x,
        'plate_z': plate_z,
        'release_speed': release_speed,
        'release_spin_rate': release_spin_rate,
        'pfx_x': pfx_x,
        'pfx_z': pfx_z,
        'release_extension': release_extension,
        'arm_angle': arm_angle,
        'balls': balls,
        'strikes': strikes,
        'stand': stand,
        'p_throws': p_throws,
        'outs_when_up': outs_when_up,
        'use_batted_model': use_batted_model,
        'bb_type': bb_type,
        'launch_speed_angle': launch_speed_angle,
        'hc_x': hc_x,
        'hc_y': hc_y
    }
