import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(page_title="clustering_music_genres", layout="centered")

st.title("Clustering Music Genres")
st.write("Adjust the audio features below to assign a song to a cluster using KMeans.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KMEANS_PATH = os.path.join(BASE_DIR, "kmeans_music_genres.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler_music_genres.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "feature_list.json")

# Load model artifacts
kmeans = joblib.load(KMEANS_PATH)
scaler = joblib.load(SCALER_PATH)

with open(FEATURES_PATH, "r") as f:
    features = json.load(f)

st.subheader("Input Audio Features")

slider_config = {
    "Beats Per Minute (BPM)": (60, 200),
    "Energy": (0, 100),
    "Danceability": (0, 100),
    "Loudness (dB)": (-30, 0),
    "Liveness": (0, 100),
    "Valence": (0, 100),
    "Acousticness": (0, 100),
    "Speechiness": (0, 100),
    "Popularity": (0, 100),
}

user_input = {}

for feat in features:
    if feat in slider_config:
        min_val, max_val = slider_config[feat]
        user_input[feat] = st.slider(feat, min_val, max_val, int((min_val + max_val)/2))
    else:
        user_input[feat] = st.number_input(feat, value=0.0)

input_df = pd.DataFrame([user_input], columns=features)

if st.button("Assign Cluster"):
    X_scaled = scaler.transform(input_df.values)
    cluster = int(kmeans.predict(X_scaled)[0])

    st.success(f"Assigned Cluster: {cluster}")
    st.write("Songs in this cluster share similar audio feature patterns.")

st.caption("Unsupervised Learning – KMeans Clustering")