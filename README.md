# clustering_music_genres (Unsupervised Learning)

This project applies KMeans clustering to group songs based on their audio features.

Unlike supervised learning, clustering identifies natural groupings in the data without predefined labels.

---

## Project Overview

- **Problem Type:** Unsupervised Learning (Clustering)
- **Algorithm:** KMeans
- **Optimal Clusters:** 5 (determined using Elbow Method)
- **Deployment:** Streamlit app on Hugging Face Spaces

---

## Dataset

Spotify-like dataset containing:

- Beats Per Minute (BPM)
- Energy
- Danceability
- Loudness (dB)
- Liveness
- Valence
- Acousticness
- Speechiness
- Popularity

Non-numerical columns (Title, Artist, Genre) were excluded from clustering.

---

## Methodology

### EDA
- Selected relevant numerical audio features
- Checked distributions and scaling needs

### Preprocessing
- Standardized features using StandardScaler

### Modeling
- Applied KMeans clustering
- Used Elbow Method to determine optimal k=5
- Reduced dimensions with PCA for visualization

---

## Deployment

The following artifacts were saved:

- `kmeans_music_genres.joblib`
- `scaler_music_genres.joblib`
- `feature_list.json`

The Streamlit app allows interactive cluster assignment.

---

## Conclusion

This project demonstrates how unsupervised learning can reveal hidden structures within music data.

Although cluster overlap exists — which is expected in real-world audio features — KMeans successfully identified meaningful groupings of songs based on acoustic similarity.

---

## Future Improvements

- Silhouette Score optimization
- Hierarchical clustering comparison
- Cluster labeling using genre distribution
