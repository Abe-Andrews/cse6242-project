import pandas as pd
import streamlit as st
import joblib
import config

@st.cache_resource
def load_pitching_model():
    model = joblib.load('models/pitch_outcome_model.joblib')
    labeler = joblib.load('models/pitch_outcome_labeler.joblib')
    return model, labeler


@st.cache_resource
def load_batting_model():
    model = joblib.load('models/batted_outcome_model.joblib')
    labeler = joblib.load('models/batted_outcome_labler.joblib')
    return model, labeler


def predict_pitch_outcome(model, labeler, user_inputs):
    inputs = dict(user_inputs)
    inputs['stand'] = 1 if inputs['stand'] == 'R' else 0
    inputs['p_throws'] = 1 if inputs['p_throws'] == 'R' else 0
    
    df = pd.DataFrame([inputs])[config.MODEL_A_COLUMNS]
    df['pitch_name'] = df['pitch_name'].astype('category')
    
    probs = model.predict_proba(df)[0]
    return dict(zip(labeler.classes_, probs))


def predict_batted_outcome(model, labeler, user_inputs, batted_inputs):
    combined = dict(user_inputs)
    combined['stand'] = 1 if combined['stand'] == 'R' else 0
    combined['p_throws'] = 1 if combined['p_throws'] == 'R' else 0
    combined.update(batted_inputs)

    df = pd.DataFrame([combined])[config.MODEL_B_COLUMNS]
    df['pitch_name'] = df['pitch_name'].astype('category')
    df['bb_type'] = df['bb_type'].astype('category')
    
    probs = model.predict_proba(df)[0]
    return dict(zip(labeler.classes_, probs))
