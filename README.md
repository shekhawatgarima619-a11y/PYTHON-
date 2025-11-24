# Music_Recommender_System
python
This project is a music recommender system built in python.It analyses user listening patterns,song attributes,and similarity metrices to recommend songs that match a user's taste.the system can be adapted for content-based,collaborative,or hybrid recommendations depending on your dataset and goals.
FEATURES
•	Content-Based Filtering: Recommends songs based on audio features, metadata, and similarity scores.
•	Collaborative Filtering (optional): Learns user behavior using interaction matrices.
•	Hybrid Approach: Combine content-based and collaborative signals.
•	Data Preprocessing Utilities: Handle missing values, scaling, and feature selection.
•	Fast Similarity Search: Cosine similarity or nearest-neighbor models.
•	Interactive Recommendation Function: Input a song or user ID to retrieve top suggestions.
Future Enhancements
	•	Add Spotify Web API for live audio extraction
	•	Train deep learning models (Autoencoders, NCF)
	•	Add mood-based playlist generation
	•	Deploy as full web app with React frontend

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(song_path, feature_path):
    songs = pd.read_csv(song_path)
    features = pd.read_csv(feature_path)
    return songs, features

def scale_features(df, feature_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    return scaled, scaler

    def __init__(self, content_model, collab_model=None):
        self.c = content_model
        self.k = collab_model

    def recommend_song(self, song, n=10):
        return self.c.recommend(song, n)

    def recommend_user(self, user_id, n=10):
        if not self.k:
            raise ValueError("Collaborative model not initialized.")
        return self.k.recommend_for_user(user_id, n)
